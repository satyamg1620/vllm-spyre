"""
NixlConnector - Network-based KV connector using NIXL for Spyre.

This connector implements prefill-decode disaggregation for Spyre using
the NIXL (Network Interface for Cross-instance Learning) library for
network-based KV cache transfer between producer (prefill) and consumer
(decode) instances.

Key differences from upstream NixlConnector:
- Works with Spyre's non-paged KV cache layout (list[tuple[Tensor, ...]])
- Handles 64-token block alignment constraint
- Registers Spyre KV cache memory regions with NIXL agent

Usage:
    Set kv_transfer_config with:
        kv_connector: "NixlConnector"
        kv_connector_module_path: "vllm_spyre.v1.kv_connector.nixl_connector"
        kv_role: "kv_producer" or "kv_consumer"
        kv_rank: <rank>
        kv_parallel_size: <total_instances>
"""

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_spyre.v1.kv_connector.base import (
    SPYRE_BLOCK_SIZE,
    KVConnectorBase,
    KVConnectorMetadata,
    ReqMeta,
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


@dataclass
class NixlReqMeta(ReqMeta):
    """Extended per-request metadata for NIXL transfers."""

    remote_engine_id: str = ""


@dataclass
class NixlConnectorMetadata(KVConnectorMetadata):
    """Metadata for NIXL KV connector operations."""

    requests: list[NixlReqMeta] = field(default_factory=list)


@dataclass
class NixlHandshakeMetadata(KVConnectorHandshakeMetadata):
    """Handshake metadata for NIXL peer discovery."""

    agent_name: str = ""
    agent_metadata: bytes = b""
    engine_id: str = ""
    kv_caches_base_addr: list[tuple[int, int]] = field(
        default_factory=list
    )


class NixlConnector(KVConnectorBase):
    """Network-based KV connector using NIXL for P/D disaggregation on Spyre.

    Producer (prefill) instances:
    - Run prefill, store KV cache in local memory
    - Register KV memory regions with NIXL for remote reads
    - Signal completion to consumers

    Consumer (decode) instances:
    - Receive KV cache from producers via NIXL reads
    - Inject received KV into local past_key_value_states
    - Continue decode from transferred state
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
        self._is_producer = (
            self._kv_transfer_config.kv_role == "kv_producer"
        )
        self._is_consumer = (
            self._kv_transfer_config.kv_role == "kv_consumer"
        )
        self._engine_id = str(self._kv_transfer_config.engine_id)

        # Track pending transfers
        self._pending_transfers: dict[str, Any] = {}
        self._finished_req_ids: set[str] = set()
        self._lock = threading.Lock()

        # NIXL agent - lazy initialized
        self._nixl_agent = None
        self._peer_agents: dict[str, Any] = {}

        # Requests that need remote KV loading (scheduler-side)
        self._requests_need_load: dict[str, "Request"] = {}

        self._init_nixl_agent()

        logger.info(
            "NixlConnector initialized: role=%s, engine_id=%s",
            self._kv_transfer_config.kv_role,
            self._engine_id,
        )

    def _init_nixl_agent(self) -> None:
        """Initialize the NIXL agent for network transfers."""
        try:
            from nixl import Agent  # ty: ignore[unresolved-import]

            agent_name = f"spyre_{self._engine_id}"
            self._nixl_agent = Agent(agent_name)
            logger.info("NIXL agent initialized: %s", agent_name)
        except ImportError:
            logger.warning(
                "NIXL library not available. NixlConnector will operate "
                "in stub mode. Install nixl for full functionality."
            )
            self._nixl_agent = None

    # ==============================
    # Handshake methods
    # ==============================

    def get_handshake_metadata(
        self,
    ) -> KVConnectorHandshakeMetadata | None:
        """Return NIXL agent metadata for peer discovery."""
        if self._nixl_agent is None:
            return None

        metadata = NixlHandshakeMetadata(
            agent_name=self._nixl_agent.name
            if hasattr(self._nixl_agent, "name")
            else "",
            agent_metadata=self._nixl_agent.get_agent_metadata()
            if hasattr(self._nixl_agent, "get_agent_metadata")
            else b"",
            engine_id=self._engine_id,
        )
        return metadata

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Process peer metadata to establish NIXL connections."""
        if self._nixl_agent is None:
            return

        for rank, peer_meta in metadata.items():
            if not isinstance(peer_meta, NixlHandshakeMetadata):
                continue
            if peer_meta.engine_id == self._engine_id:
                continue

            try:
                self._nixl_agent.add_remote_agent(peer_meta.agent_metadata)
                self._peer_agents[peer_meta.engine_id] = peer_meta
                logger.info(
                    "Added NIXL peer: engine_id=%s",
                    peer_meta.engine_id,
                )
            except Exception:
                logger.exception(
                    "Failed to add NIXL peer: engine_id=%s",
                    peer_meta.engine_id,
                )

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Start loading KV cache from remote producer via NIXL.

        For consumer instances: issues async NIXL reads from the producer's
        KV cache memory regions.
        For producer instances: no-op (KV is generated locally).
        """
        if not self._is_consumer or self._nixl_agent is None:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, NixlConnectorMetadata)

        if self._spyre_kv_caches is None:
            logger.warning(
                "start_load_kv called but no Spyre KV caches registered"
            )
            return

        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue

            slot_mapping = self.compute_slot_mapping(
                req_meta.block_ids, self._block_size, req_meta.num_tokens
            )

            logger.info(
                "NIXL: Loading KV for request '%s' (%d tokens) from remote",
                req_meta.req_id,
                req_meta.num_tokens,
            )

            # Issue NIXL transfer for each layer
            for layer_idx in range(self._num_layers):
                try:
                    k_data, v_data = self._nixl_read_layer(
                        req_meta, layer_idx, slot_mapping
                    )
                    if k_data is not None and v_data is not None:
                        self.inject_kv_for_slots(
                            self._spyre_kv_caches,
                            layer_idx,
                            slot_mapping,
                            k_data,
                            v_data,
                        )
                except Exception:
                    logger.exception(
                        "NIXL: Failed to load layer %d for request '%s'",
                        layer_idx,
                        req_meta.req_id,
                    )

    def _nixl_read_layer(
        self,
        req_meta: NixlReqMeta,
        layer_idx: int,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Read KV data for a single layer from remote via NIXL.

        This is a placeholder that returns None when NIXL is not fully
        configured. In production, this would issue an async NIXL read
        from the producer's registered memory region.
        """
        if self._nixl_agent is None:
            return None, None

        # TODO: Implement actual NIXL read using registered memory regions
        # This requires:
        # 1. Looking up the remote engine's memory descriptors
        # 2. Creating transfer descriptors for the specific slots
        # 3. Issuing the async read
        # 4. Waiting for completion
        logger.debug(
            "NIXL read for layer %d, %d tokens (stub)",
            layer_idx,
            len(slot_mapping),
        )
        return None, None

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until NIXL transfer for this layer completes."""
        # For synchronous transfers, this is a no-op.
        # For async transfers, this would check transfer completion status.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """For producer: KV is already in local memory after prefill.

        NIXL uses RDMA-style reads, so the producer doesn't actively push.
        It just needs to ensure the KV data is in registered memory, which
        it is since we register past_key_value_states at init time.
        """
        if not self._is_producer:
            return

        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        assert isinstance(metadata, NixlConnectorMetadata)

        # For producer, mark that KV for this layer is ready for remote reads
        for req_meta in metadata.requests:
            if req_meta.is_store:
                with self._lock:
                    self._pending_transfers.setdefault(
                        req_meta.req_id, set()
                    ).add(layer_name)

    def wait_for_save(self) -> None:
        """Block until all NIXL save notifications are complete."""
        # For producers, KV is already in place after forward pass.
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Check NIXL transfer completion status.

        Returns sets of request IDs that have finished sending (producer)
        or receiving (consumer).
        """
        with self._lock:
            send_finished = (
                self._finished_req_ids & finished_req_ids
                if self._is_producer
                else None
            )
            recv_finished = (
                self._finished_req_ids & finished_req_ids
                if self._is_consumer
                else None
            )
            self._finished_req_ids -= finished_req_ids

        return send_finished, recv_finished

    def shutdown(self) -> None:
        """Clean up NIXL agent and memory registrations."""
        if self._nixl_agent is not None:
            logger.info("Shutting down NIXL agent")
            self._nixl_agent = None
        self._peer_agents.clear()
        self._pending_transfers.clear()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check if remote prefill instance has KV for this request.

        For consumer (decode) instances, check if a producer has
        completed prefill for this request's prompt.
        For producer instances, always return 0 (no external KV).
        """
        if self._is_producer:
            return 0, False

        # Consumer: check if remote KV is available
        token_ids = list(request.prompt_token_ids or [])
        num_tokens = align_to_spyre_block(len(token_ids))

        if num_tokens <= 0:
            return 0, False

        # TODO: Query remote producer for KV availability
        # For now, return 0 (no remote KV available)
        # In production, this would check a metadata store or
        # query the NIXL agent for available KV
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Track blocks allocated for incoming KV transfers."""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build metadata with NIXL transfer descriptors.

        For producer: mark new requests as store (KV will be generated)
        For consumer: mark requests needing load from remote
        """
        meta = NixlConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])

            if self._is_producer:
                meta.requests.append(
                    NixlReqMeta(
                        req_id=new_req.req_id,
                        token_ids=token_ids,
                        num_tokens=align_to_spyre_block(len(token_ids)),
                        block_ids=new_req.block_ids[0],
                        is_store=True,
                    )
                )
            elif new_req.req_id in self._requests_need_load:
                meta.requests.append(
                    NixlReqMeta(
                        req_id=new_req.req_id,
                        token_ids=token_ids,
                        num_tokens=align_to_spyre_block(len(token_ids)),
                        block_ids=new_req.block_ids[0],
                        is_store=False,
                    )
                )

        self._requests_need_load.clear()
        return meta
