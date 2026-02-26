"""Tests for OffloadingConnector (CPU memory offloading)."""

import shutil
import tempfile
from unittest.mock import MagicMock

import pytest
import torch

from vllm_spyre.v1.kv_connector.base import (
    SPYRE_BLOCK_SIZE,
    KVConnectorBase,
    KVConnectorMetadata,
)
from vllm_spyre.v1.kv_connector.offloading_connector import (
    OffloadingConnector,
    _CPUKVEntry,
)


def _make_mock_vllm_config(max_entries=100):
    """Create a mock VllmConfig for testing."""
    config = MagicMock()
    config.cache_config.block_size = SPYRE_BLOCK_SIZE
    config.kv_transfer_config.get_from_extra_config.return_value = max_entries
    config.kv_transfer_config.kv_connector = "OffloadingConnector"
    config.kv_transfer_config.kv_role = "kv_both"
    config.kv_transfer_config.kv_rank = 0
    config.kv_transfer_config.kv_parallel_size = 1
    config.kv_transfer_config.engine_id = "test-engine"
    return config


def _make_kv_states(num_layers=2, num_blocks=4, block_size=64,
                    num_kv_heads=4, head_dim=32):
    """Create dummy KV states in Spyre format."""
    kv_states = []
    for _ in range(num_layers):
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        kv_states.append((k, v))
    return kv_states


class TestCPUKVEntry:
    """Tests for _CPUKVEntry dataclass."""

    def test_create_entry(self):
        kv_data = [
            (torch.randn(64, 4, 32), torch.randn(64, 4, 32))
            for _ in range(2)
        ]
        entry = _CPUKVEntry(kv_data=kv_data, num_tokens=64, num_layers=2)
        assert entry.num_tokens == 64
        assert entry.num_layers == 2
        assert len(entry.kv_data) == 2


class TestOffloadingConnectorInit:
    """Tests for connector initialization."""

    def test_init(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config()
        connector = OffloadingConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        assert connector._block_size == SPYRE_BLOCK_SIZE
        assert len(connector._cpu_cache) == 0


class TestOffloadingConnectorSaveLoad:
    """Tests for CPU offload save and load operations."""

    @pytest.fixture
    def connector(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config()
        conn = OffloadingConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states = _make_kv_states()
        conn.register_spyre_kv_caches(kv_states)
        return conn

    def test_save_populates_cpu_cache(self, connector):
        """Saving KV should create an entry in the CPU cache."""
        token_ids = list(range(64))
        meta = KVConnectorMetadata()
        meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=[1],
            is_store=True,
        )
        connector.bind_connector_metadata(meta)

        for layer_idx in range(2):
            connector.save_kv_layer(
                layer_name=f"layer_{layer_idx}",
                kv_layer=torch.empty(0),
                attn_metadata=MagicMock(),
            )
        connector.wait_for_save()

        assert len(connector._cpu_cache) == 1

    def test_save_load_roundtrip(self):
        """Test that save to CPU and load back produces identical data."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config()

        # Create connector with known KV data
        connector = OffloadingConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states = _make_kv_states(num_layers=2)
        connector.register_spyre_kv_caches(kv_states)

        token_ids = list(range(64))
        block_ids = [1]

        # Save to CPU
        save_meta = KVConnectorMetadata()
        save_meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=block_ids,
            is_store=True,
        )
        connector.bind_connector_metadata(save_meta)

        for layer_idx in range(2):
            connector.save_kv_layer(
                layer_name=f"layer_{layer_idx}",
                kv_layer=torch.empty(0),
                attn_metadata=MagicMock(),
            )
        connector.wait_for_save()
        connector.clear_connector_metadata()

        # Extract original data for comparison
        slot_mapping = KVConnectorBase.compute_slot_mapping(
            block_ids=block_ids, block_size=SPYRE_BLOCK_SIZE, num_tokens=64
        )
        originals = []
        for layer_idx in range(2):
            k, v = KVConnectorBase.extract_kv_for_slots(
                kv_states, layer_idx, slot_mapping
            )
            originals.append((k.clone(), v.clone()))

        # Zero out the block we'll load into
        for layer in kv_states:
            layer[0][1].zero_()
            layer[1][1].zero_()

        # Load from CPU
        load_meta = KVConnectorMetadata()
        load_meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=block_ids,
            is_store=False,
        )
        connector.bind_connector_metadata(load_meta)

        forward_ctx = MagicMock()
        connector.start_load_kv(forward_ctx)

        # Verify roundtrip
        for layer_idx in range(2):
            k_loaded, v_loaded = KVConnectorBase.extract_kv_for_slots(
                kv_states, layer_idx, slot_mapping
            )
            k_orig, v_orig = originals[layer_idx]
            assert torch.allclose(k_orig, k_loaded), (
                f"Layer {layer_idx} key mismatch"
            )
            assert torch.allclose(v_orig, v_loaded), (
                f"Layer {layer_idx} value mismatch"
            )


class TestOffloadingConnectorEviction:
    """Tests for CPU cache eviction."""

    def test_eviction_when_max_entries_exceeded(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(max_entries=2)
        connector = OffloadingConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states = _make_kv_states()
        connector.register_spyre_kv_caches(kv_states)

        # Save 3 entries (max is 2)
        for i in range(3):
            token_ids = list(range(i * 64, (i + 1) * 64))
            meta = KVConnectorMetadata()
            meta.add_new_request(
                req_id=f"req-{i}",
                token_ids=token_ids,
                block_ids=[1],
                is_store=True,
            )
            connector.bind_connector_metadata(meta)

            for layer_idx in range(2):
                connector.save_kv_layer(
                    layer_name=f"layer_{layer_idx}",
                    kv_layer=torch.empty(0),
                    attn_metadata=MagicMock(),
                )
            connector.wait_for_save()
            connector.clear_connector_metadata()

        # Should have evicted down to max_entries
        assert len(connector._cpu_cache) <= 2


class TestOffloadingConnectorScheduler:
    """Tests for scheduler-side methods."""

    @pytest.fixture
    def scheduler_connector(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config()
        return OffloadingConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
        )

    def test_get_num_matched_tokens_no_cache(self, scheduler_connector):
        """Should return 0 when CPU cache is empty."""
        request = MagicMock()
        request.prompt_token_ids = list(range(128))
        request.request_id = "req-1"

        num_matched, is_async = scheduler_connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )
        assert num_matched == 0
        assert is_async is False

    def test_build_connector_meta_store(self, scheduler_connector):
        """Should mark new requests as store."""
        scheduler_output = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "req-1"
        new_req.prompt_token_ids = list(range(128))
        new_req.block_ids = [[0, 1]]
        scheduler_output.scheduled_new_reqs = [new_req]

        meta = scheduler_connector.build_connector_meta(scheduler_output)
        assert isinstance(meta, KVConnectorMetadata)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True

    def test_request_finished_returns_false(self, scheduler_connector):
        """request_finished should return False (sync saves)."""
        request = MagicMock()
        request.request_id = "req-1"

        should_keep, params = scheduler_connector.request_finished(
            request, block_ids=[0, 1]
        )
        assert should_keep is False
        assert params is None
