"""Tests for SpyreKVConnectorBase and utility functions."""

import pytest
import torch

from vllm_spyre.v1.kv_connector.base import (
    SPYRE_BLOCK_SIZE,
    SpyreKVConnectorBase,
    SpyreKVConnectorMetadata,
    SpyreReqMeta,
    align_to_spyre_block,
)


class TestAlignToSpyreBlock:
    """Tests for align_to_spyre_block utility function."""

    def test_exact_multiple(self):
        assert align_to_spyre_block(64) == 64
        assert align_to_spyre_block(128) == 128
        assert align_to_spyre_block(256) == 256

    def test_rounds_down(self):
        assert align_to_spyre_block(65) == 64
        assert align_to_spyre_block(100) == 64
        assert align_to_spyre_block(127) == 64
        assert align_to_spyre_block(129) == 128

    def test_zero(self):
        assert align_to_spyre_block(0) == 0

    def test_less_than_block_size(self):
        assert align_to_spyre_block(1) == 0
        assert align_to_spyre_block(63) == 0

    def test_large_values(self):
        assert align_to_spyre_block(8192) == 8192
        assert align_to_spyre_block(8193) == 8192


class TestSpyreReqMeta:
    """Tests for SpyreReqMeta dataclass."""

    def test_create_req_meta(self):
        meta = SpyreReqMeta(
            req_id="req-1",
            token_ids=[1, 2, 3, 4],
            num_tokens=4,
            block_ids=[0, 1],
            is_store=True,
        )
        assert meta.req_id == "req-1"
        assert meta.num_tokens == 4
        assert meta.is_store is True
        assert len(meta.block_ids) == 2


class TestSpyreKVConnectorMetadata:
    """Tests for SpyreKVConnectorMetadata."""

    def test_empty_metadata(self):
        meta = SpyreKVConnectorMetadata()
        assert len(meta.requests) == 0

    def test_add_new_request(self):
        meta = SpyreKVConnectorMetadata()
        token_ids = list(range(128))
        meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=[0, 1],
            is_store=True,
        )
        assert len(meta.requests) == 1
        req = meta.requests[0]
        assert req.req_id == "req-1"
        assert req.is_store is True
        assert req.num_tokens == 128  # 128 is aligned to 64

    def test_add_request_aligns_tokens(self):
        meta = SpyreKVConnectorMetadata()
        # 100 tokens should be aligned down to 64
        token_ids = list(range(100))
        meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=[0],
            is_store=False,
        )
        assert meta.requests[0].num_tokens == 64

    def test_add_multiple_requests(self):
        meta = SpyreKVConnectorMetadata()
        for i in range(3):
            meta.add_new_request(
                req_id=f"req-{i}",
                token_ids=list(range(64)),
                block_ids=[i],
                is_store=i % 2 == 0,
            )
        assert len(meta.requests) == 3
        assert meta.requests[0].is_store is True
        assert meta.requests[1].is_store is False


class TestComputeSlotMapping:
    """Tests for SpyreKVConnectorBase.compute_slot_mapping."""

    def test_single_block(self):
        slot_mapping = SpyreKVConnectorBase.compute_slot_mapping(
            block_ids=[2], block_size=64, num_tokens=64
        )
        assert slot_mapping.shape == (64,)
        # Block 2 -> slots 128..191
        assert slot_mapping[0].item() == 128
        assert slot_mapping[63].item() == 191

    def test_multiple_blocks(self):
        slot_mapping = SpyreKVConnectorBase.compute_slot_mapping(
            block_ids=[0, 3], block_size=64, num_tokens=128
        )
        assert slot_mapping.shape == (128,)
        # Block 0 -> slots 0..63
        assert slot_mapping[0].item() == 0
        assert slot_mapping[63].item() == 63
        # Block 3 -> slots 192..255
        assert slot_mapping[64].item() == 192
        assert slot_mapping[127].item() == 255

    def test_truncated_to_num_tokens(self):
        slot_mapping = SpyreKVConnectorBase.compute_slot_mapping(
            block_ids=[1], block_size=64, num_tokens=32
        )
        assert slot_mapping.shape == (32,)
        assert slot_mapping[0].item() == 64
        assert slot_mapping[31].item() == 95


class TestExtractInjectKV:
    """Tests for extract_kv_for_slots and inject_kv_for_slots."""

    def _make_kv_states(self, num_layers=2, num_blocks=4,
                        block_size=64, num_kv_heads=4, head_dim=32):
        """Create dummy KV states in Spyre format."""
        kv_states = []
        for layer in range(num_layers):
            k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
            v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
            kv_states.append((k, v))
        return kv_states

    def test_extract_kv(self):
        kv_states = self._make_kv_states()
        slot_indices = torch.tensor([0, 1, 64, 65])

        k_data, v_data = SpyreKVConnectorBase.extract_kv_for_slots(
            kv_states, layer_idx=0, slot_indices=slot_indices
        )

        assert k_data.shape == (4, 4, 32)
        assert v_data.shape == (4, 4, 32)
        # Verify data matches the original
        k_flat = kv_states[0][0].reshape(-1, 4, 32)
        assert torch.equal(k_data[0], k_flat[0])
        assert torch.equal(k_data[2], k_flat[64])

    def test_inject_kv(self):
        kv_states = self._make_kv_states()
        slot_indices = torch.tensor([0, 1])
        k_data = torch.ones(2, 4, 32)
        v_data = torch.ones(2, 4, 32) * 2

        SpyreKVConnectorBase.inject_kv_for_slots(
            kv_states, layer_idx=0, slot_indices=slot_indices,
            k_data=k_data, v_data=v_data
        )

        # Verify injection
        k_flat = kv_states[0][0].reshape(-1, 4, 32)
        v_flat = kv_states[0][1].reshape(-1, 4, 32)
        assert torch.equal(k_flat[0], torch.ones(4, 32))
        assert torch.equal(v_flat[0], torch.ones(4, 32) * 2)
        assert torch.equal(k_flat[1], torch.ones(4, 32))

    def test_extract_inject_roundtrip(self):
        """Verify that extracting and re-injecting preserves data."""
        kv_states = self._make_kv_states()
        slot_indices = torch.tensor([10, 20, 30])

        # Extract
        k_orig, v_orig = SpyreKVConnectorBase.extract_kv_for_slots(
            kv_states, layer_idx=1, slot_indices=slot_indices
        )

        # Zero out those slots
        SpyreKVConnectorBase.inject_kv_for_slots(
            kv_states, layer_idx=1, slot_indices=slot_indices,
            k_data=torch.zeros_like(k_orig),
            v_data=torch.zeros_like(v_orig),
        )

        # Re-inject original data
        SpyreKVConnectorBase.inject_kv_for_slots(
            kv_states, layer_idx=1, slot_indices=slot_indices,
            k_data=k_orig, v_data=v_orig,
        )

        # Verify roundtrip
        k_check, v_check = SpyreKVConnectorBase.extract_kv_for_slots(
            kv_states, layer_idx=1, slot_indices=slot_indices
        )
        assert torch.equal(k_orig, k_check)
        assert torch.equal(v_orig, v_check)
