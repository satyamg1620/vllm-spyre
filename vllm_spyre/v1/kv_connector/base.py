"""
KVConnectorBase - Base class for Spyre-specific KV connectors.

Spyre's KV cache layout differs from upstream vLLM's paged attention:
- KV cache is `list[tuple[Tensor, Tensor]]` per layer
- Each tensor has shape [num_blocks, block_size, num_kv_heads, head_dim]
- Block size is fixed at 64 tokens (hardware constraint)
- No paged attention - block_size = max_model_len for vLLM scheduler

This base class provides helpers to work with Spyre's KV cache layout
while implementing the upstream KVConnectorBase_V1 interface.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata as _BaseKVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Spyre hardware block alignment constraint
SPYRE_BLOCK_SIZE = 64


@dataclass
class ReqMeta:
    """Per-request metadata for Spyre KV connector operations."""

    req_id: str
    token_ids: list[int]
    num_tokens: int
    block_ids: list[int]
    is_store: bool


@dataclass
class KVConnectorMetadata(_BaseKVConnectorMetadata):
    """Metadata for Spyre KV connector operations.

    Passed from the scheduler-side connector to the worker-side connector
    via the SchedulerOutput.
    """

    requests: list[ReqMeta] = field(default_factory=list)

    def add_new_request(
        self,
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
        is_store: bool,
    ) -> None:
        num_tokens = align_to_spyre_block(len(token_ids))
        self.requests.append(
            ReqMeta(
                req_id=req_id,
                token_ids=token_ids[:num_tokens],
                num_tokens=num_tokens,
                block_ids=block_ids,
                is_store=is_store,
            )
        )


class KVConnectorBase(KVConnectorBase_V1):
    """Base class for Spyre-specific KV connectors.

    Extends KVConnectorBase_V1 with helpers for Spyre's unique KV cache
    layout (list of tuples of tensors per layer, not paged tensors).
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
        self._spyre_kv_caches: (
            list[tuple[torch.Tensor, torch.Tensor]] | None
        ) = None
        self._num_layers: int = 0

    def register_spyre_kv_caches(
        self,
        past_key_value_states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Register reference to SpyreCausalLM's KV cache tensors.

        This is called after model loading to give the connector direct
        access to the model's KV cache buffers.

        Args:
            past_key_value_states: list of (key, value) tensor tuples per layer.
                Each tensor has shape [num_blocks, block_size, num_kv_heads, head_dim].
        """
        self._spyre_kv_caches = past_key_value_states
        self._num_layers = len(past_key_value_states)
        logger.info(
            "Registered Spyre KV caches: %d layers", self._num_layers
        )

    @staticmethod
    def extract_kv_for_slots(
        kv_states: list[tuple[torch.Tensor, ...]],
        layer_idx: int,
        slot_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract KV data for specific slot positions from a layer.

        Spyre KV cache tensors have shape:
            [num_blocks, block_size, num_kv_heads, head_dim]
        We flatten the first two dims to get [total_slots, num_kv_heads, head_dim]
        then index by slot_indices.

        Args:
            kv_states: The model's past_key_value_states.
            layer_idx: Which layer to extract from.
            slot_indices: 1-D tensor of slot positions to extract.

        Returns:
            Tuple of (k_data, v_data), each with shape
            [num_slots, num_kv_heads, head_dim].
        """
        k_tensor = kv_states[layer_idx][0]
        v_tensor = kv_states[layer_idx][1]
        # Flatten [num_blocks, block_size, ...] -> [total_slots, ...]
        k_flat = k_tensor.reshape(-1, *k_tensor.shape[2:])
        v_flat = v_tensor.reshape(-1, *v_tensor.shape[2:])
        return k_flat[slot_indices], v_flat[slot_indices]

    @staticmethod
    def inject_kv_for_slots(
        kv_states: list[tuple[torch.Tensor, ...]],
        layer_idx: int,
        slot_indices: torch.Tensor,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
    ) -> None:
        """Inject KV data into specific slot positions in a layer.

        Args:
            kv_states: The model's past_key_value_states.
            layer_idx: Which layer to inject into.
            slot_indices: 1-D tensor of slot positions.
            k_data: Key data with shape [num_slots, num_kv_heads, head_dim].
            v_data: Value data with shape [num_slots, num_kv_heads, head_dim].
        """
        k_tensor = kv_states[layer_idx][0]
        v_tensor = kv_states[layer_idx][1]
        k_flat = k_tensor.reshape(-1, *k_tensor.shape[2:])
        v_flat = v_tensor.reshape(-1, *v_tensor.shape[2:])
        k_flat[slot_indices] = k_data
        v_flat[slot_indices] = v_data

    @staticmethod
    def compute_slot_mapping(
        block_ids: list[int], block_size: int, num_tokens: int
    ) -> torch.Tensor:
        """Compute slot mapping from block IDs.

        Maps tokens to their physical positions in the KV cache buffer.

        Args:
            block_ids: List of block IDs allocated for this request.
            block_size: Number of slots per block.
            num_tokens: Number of tokens to map.

        Returns:
            1-D tensor of slot indices with length num_tokens.
        """
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.int64)
        num_blocks = len(block_ids)
        block_offsets = torch.arange(block_size, dtype=torch.int64)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids_tensor.reshape(num_blocks, 1) * block_size
        )
        return slot_mapping.flatten()[:num_tokens]


def align_to_spyre_block(num_tokens: int) -> int:
    """Align token count down to the Spyre 64-token block boundary.

    Args:
        num_tokens: Number of tokens.

    Returns:
        Largest multiple of SPYRE_BLOCK_SIZE that is <= num_tokens.
    """
    return (num_tokens // SPYRE_BLOCK_SIZE) * SPYRE_BLOCK_SIZE
