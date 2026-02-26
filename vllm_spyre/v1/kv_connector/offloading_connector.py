"""
OffloadingConnector - CPU memory offloading KV connector for Spyre.

This connector offloads KV cache from Spyre device memory to CPU memory
for reuse across requests with matching prompts. This enables prefix
caching at the KV cache level without requiring device memory to hold
all cached entries.

Key operations:
- After a request finishes, KV cache is copied to CPU buffers
- When a new request matches a cached prefix, KV is restored from CPU
- Preempted requests' KV is saved to CPU before blocks are freed

This is adapted for Spyre's non-paged KV cache layout.
"""

import threading
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_spyre.v1.kv_connector.base import (
    SPYRE_BLOCK_SIZE,
    KVConnectorBase,
    KVConnectorMetadata,
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


class _CPUKVEntry:
    """A single CPU-side KV cache entry for a specific prompt prefix."""

    __slots__ = ("kv_data", "num_tokens", "num_layers")

    def __init__(
        self,
        kv_data: list[tuple[torch.Tensor, torch.Tensor]],
        num_tokens: int,
        num_layers: int,
    ):
        self.kv_data = kv_data
        self.num_tokens = num_tokens
        self.num_layers = num_layers


class OffloadingConnector(KVConnectorBase):
    """Offloads KV cache to CPU memory for later reuse across requests.

    Maintains a dictionary mapping prompt hashes to CPU-side KV cache
    tensors. When a new request has a matching prefix, the cached KV
    is restored to the device.

    Configuration via kv_transfer_config extra_config:
        max_cpu_cache_entries: Maximum number of entries to cache (default: 100)
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
        self._max_entries = self._kv_transfer_config.get_from_extra_config(
            "max_cpu_cache_entries", 100
        )

        # CPU-side KV cache: hash -> _CPUKVEntry
        self._cpu_cache: dict[str, _CPUKVEntry] = {}
        self._lock = threading.Lock()

        # Requests that need KV loading from CPU (scheduler-side)
        self._requests_need_load: dict[str, "Request"] = {}

        logger.info(
            "OffloadingConnector initialized: max_entries=%d",
            self._max_entries,
        )

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Restore KV cache from CPU buffers into device KV cache.

        For requests with matching cached prefixes, copies the KV data
        from CPU memory into the model's past_key_value_states.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, KVConnectorMetadata)

        if self._spyre_kv_caches is None:
            logger.warning(
                "start_load_kv called but no Spyre KV caches registered"
            )
            return

        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue

            cache_key = self._compute_cache_key(
                req_meta.token_ids[:req_meta.num_tokens]
            )

            with self._lock:
                entry = self._cpu_cache.get(cache_key)

            if entry is None:
                logger.warning(
                    "CPU cache miss for request '%s' (expected hit)",
                    req_meta.req_id,
                )
                continue

            slot_mapping = self.compute_slot_mapping(
                req_meta.block_ids, self._block_size, req_meta.num_tokens
            )

            logger.info(
                "Restoring KV from CPU cache for request '%s': %d tokens",
                req_meta.req_id,
                req_meta.num_tokens,
            )

            for layer_idx in range(entry.num_layers):
                k_data, v_data = entry.kv_data[layer_idx]
                # Copy from CPU to device
                self.inject_kv_for_slots(
                    self._spyre_kv_caches,
                    layer_idx,
                    slot_mapping,
                    k_data.to(self._spyre_kv_caches[layer_idx][0].device),
                    v_data.to(self._spyre_kv_caches[layer_idx][1].device),
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until CPU->device copy for this layer completes.

        CPU->device copies are synchronous for now, so this is a no-op.
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Save KV for a layer to CPU buffers for finished requests.

        This extracts KV data from the model's past_key_value_states
        and stores it in CPU memory for later reuse.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, KVConnectorMetadata)

        if self._spyre_kv_caches is None:
            return

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

            cache_key = self._compute_cache_key(
                req_meta.token_ids[:req_meta.num_tokens]
            )

            with self._lock:
                if cache_key not in self._cpu_cache:
                    # Create a new entry with placeholder for all layers
                    self._cpu_cache[cache_key] = _CPUKVEntry(
                        kv_data=[
                            (torch.empty(0), torch.empty(0))
                            for _ in range(self._num_layers)
                        ],
                        num_tokens=req_meta.num_tokens,
                        num_layers=self._num_layers,
                    )

                entry = self._cpu_cache[cache_key]
                entry.kv_data[layer_idx] = (
                    k_data.detach().cpu(),
                    v_data.detach().cpu(),
                )

    def wait_for_save(self) -> None:
        """Block until all device->CPU copies complete.

        Copies are synchronous, so this is a no-op. Also enforces
        the max cache size by evicting old entries.
        """
        self._evict_if_needed()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check CPU cache for KV matching this request's prompt.

        If a matching prefix exists in CPU memory, returns the number
        of additional tokens that can be loaded from the cache.
        """
        token_ids = list(request.prompt_token_ids or [])
        # len - 1: exclude the last token (needs forward pass to compute KV)
        num_tokens = align_to_spyre_block(len(token_ids) - 1)

        if num_tokens <= 0:
            return 0, False

        cache_key = self._compute_cache_key(token_ids[:num_tokens])

        with self._lock:
            entry = self._cpu_cache.get(cache_key)

        if entry is None:
            return 0, False

        logger.info(
            "CPU cache hit for request '%s': %d tokens",
            request.request_id,
            entry.num_tokens,
        )
        return entry.num_tokens - num_computed_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Track requests needing CPU->device KV restoration."""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build metadata for CPU offload/reload operations.

        For new requests:
        - If request needs loading from CPU cache, mark as load
        - Otherwise, mark as store (KV will be saved to CPU after completion)
        """
        meta = KVConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])

            if new_req.req_id in self._requests_need_load:
                meta.add_new_request(
                    req_id=new_req.req_id,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    is_store=False,
                )
            else:
                # Store for future reuse
                meta.add_new_request(
                    req_id=new_req.req_id,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    is_store=True,
                )

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes.

        KV is saved synchronously during save_kv_layer, so blocks can
        be freed immediately (return False).
        """
        return False, None

    # ==============================
    # Helper methods
    # ==============================

    @staticmethod
    def _compute_cache_key(token_ids: list[int]) -> str:
        """Compute a hash key for a sequence of token IDs."""
        token_bytes = torch.tensor(token_ids).numpy().tobytes()
        return safe_hash(token_bytes, usedforsecurity=False).hexdigest()

    def _evict_if_needed(self) -> None:
        """Evict oldest cache entries if we exceed max_entries."""
        with self._lock:
            while len(self._cpu_cache) > self._max_entries:
                # Evict the first (oldest) entry
                oldest_key = next(iter(self._cpu_cache))
                del self._cpu_cache[oldest_key]
                logger.debug("Evicted CPU cache entry: %s", oldest_key)
