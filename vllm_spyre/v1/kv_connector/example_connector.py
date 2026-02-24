"""
SpyreExampleConnector - Disk-based debug KV connector for Spyre.

This connector saves/loads KV cache to/from disk using safetensors files.
It is adapted from the upstream vLLM ExampleConnector to work with Spyre's
non-paged KV cache layout (list of tuples of tensors per layer).

This connector is primarily useful for:
- Debugging KV cache transfer logic
- Testing disaggregated serving without network dependencies
- Validating that KV cache save/load produces correct results
"""

import os
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_spyre.v1.kv_connector.base import (
    SPYRE_BLOCK_SIZE,
    SpyreKVConnectorBase,
    SpyreKVConnectorMetadata,
    align_to_spyre_block,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class SpyreExampleConnector(SpyreKVConnectorBase):
    """Disk-based debug connector for Spyre KV cache save/load.

    Saves KV cache to safetensors files on disk, keyed by a hash of
    the prompt token IDs. Loads KV cache back from disk when a matching
    prompt is detected.

    Configuration via kv_transfer_config extra_config:
        shared_storage_path: Directory for KV cache files (default: /tmp)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = SPYRE_BLOCK_SIZE
        self._requests_need_load: dict[str, "Request"] = {}
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        logger.info(
            "SpyreExampleConnector initialized with storage_path=%s",
            self._storage_path,
        )

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Load KV cache from disk into Spyre's past_key_value_states.

        For each load request in metadata, reads safetensors files
        (one per layer) and injects the KV data into the model's
        KV cache buffers at the correct slot positions.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SpyreKVConnectorMetadata)

        if self._spyre_kv_caches is None:
            logger.warning(
                "start_load_kv called but no Spyre KV caches registered"
            )
            return

        try:
            import safetensors.torch
        except ImportError:
            logger.error("safetensors is required for SpyreExampleConnector")
            return

        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue

            slot_mapping = self.compute_slot_mapping(
                req_meta.block_ids, self._block_size, req_meta.num_tokens
            )

            logger.info(
                "Loading KV cache for request '%s': %d tokens",
                req_meta.req_id,
                req_meta.num_tokens,
            )

            for layer_idx in range(self._num_layers):
                layer_name = f"layer_{layer_idx}"
                filename = self._generate_filename(
                    layer_name, req_meta.token_ids[:req_meta.num_tokens]
                )
                if not os.path.exists(filename):
                    logger.warning(
                        "KV cache file not found: %s", filename
                    )
                    continue

                kv_data = safetensors.torch.load_file(filename)
                k_data = kv_data["k"]
                v_data = kv_data["v"]
                self.inject_kv_for_slots(
                    self._spyre_kv_caches,
                    layer_idx,
                    slot_mapping,
                    k_data,
                    v_data,
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op: disk loads are synchronous."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Save KV cache for a specific layer to disk.

        This method is called per-layer during/after the forward pass.
        For Spyre, we extract KV data from the model's past_key_value_states
        directly (since kv_layer from upstream is not applicable to Spyre's
        layout).

        The layer_name should be in the format "layer_{idx}".
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SpyreKVConnectorMetadata)

        if self._spyre_kv_caches is None:
            return

        try:
            import safetensors.torch
        except ImportError:
            logger.error("safetensors is required for SpyreExampleConnector")
            return

        # Parse layer index from layer_name
        layer_idx = int(layer_name.split("_")[-1])

        for req_meta in metadata.requests:
            if not req_meta.is_store:
                continue

            slot_mapping = self.compute_slot_mapping(
                req_meta.block_ids, self._block_size, req_meta.num_tokens
            )

            k_data, v_data = self.extract_kv_for_slots(
                self._spyre_kv_caches, layer_idx, slot_mapping
            )

            filename = self._generate_filename(
                layer_name, req_meta.token_ids[:req_meta.num_tokens]
            )
            tensors = {
                "k": k_data.detach().cpu(),
                "v": v_data.detach().cpu(),
            }
            safetensors.torch.save_file(tensors, filename)

    def wait_for_save(self) -> None:
        """No-op: disk saves are synchronous."""
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check if KV cache exists on disk for this request's prompt.

        Returns the number of additional tokens that can be loaded from
        the external KV cache beyond what is already computed.
        """
        token_ids = list(request.prompt_token_ids or [])
        if not self._found_match_for_prompt(token_ids):
            return 0, False

        logger.info("External cache hit for request '%s'", request.request_id)

        num_tokens_to_check = align_to_spyre_block(len(token_ids) - 1)
        return num_tokens_to_check - num_computed_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Track requests that need KV loading after block allocation."""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build metadata telling the worker which requests to load/store.

        For new requests:
        - If the request needs loading (was in _requests_need_load), mark as load
        - If no cached KV exists on disk, mark as store
        - Otherwise, skip (already cached)

        Resets _requests_need_load after building metadata.
        """
        meta = SpyreKVConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            if new_req.req_id in self._requests_need_load:
                meta.add_new_request(
                    req_id=new_req.req_id,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    is_store=False,
                )
            elif not self._found_match_for_prompt(token_ids):
                meta.add_new_request(
                    req_id=new_req.req_id,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    is_store=True,
                )

        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helper methods
    # ==============================

    def _found_match_for_prompt(self, prompt_token_ids: list[int]) -> bool:
        """Check if KV cache files exist on disk for this prompt."""
        num_tokens = align_to_spyre_block(len(prompt_token_ids) - 1)
        if num_tokens <= 0:
            return False
        foldername = self._generate_foldername(
            prompt_token_ids[:num_tokens], create_folder=False
        )
        return os.path.exists(foldername)

    def _generate_foldername(
        self, token_ids: list[int], create_folder: bool = False
    ) -> str:
        """Generate a folder name based on the hash of the token IDs."""
        token_bytes = torch.tensor(token_ids).numpy().tobytes()
        input_ids_hash = safe_hash(
            token_bytes, usedforsecurity=False
        ).hexdigest()
        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename(
        self, layer_name: str, token_ids: list[int]
    ) -> str:
        """Generate a file path for a specific layer's KV cache."""
        foldername = self._generate_foldername(token_ids, create_folder=True)
        return os.path.join(foldername, f"{layer_name}.safetensors")
