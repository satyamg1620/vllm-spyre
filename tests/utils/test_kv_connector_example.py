"""Tests for SpyreExampleConnector (disk-based debug connector)."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_spyre.v1.kv_connector.base import (
    SPYRE_BLOCK_SIZE,
    SpyreKVConnectorBase,
    SpyreKVConnectorMetadata,
    align_to_spyre_block,
)
from vllm_spyre.v1.kv_connector.example_connector import SpyreExampleConnector


def _make_mock_vllm_config(storage_path="/tmp"):
    """Create a mock VllmConfig for testing."""
    config = MagicMock()
    config.cache_config.block_size = SPYRE_BLOCK_SIZE
    config.kv_transfer_config.get_from_extra_config.return_value = storage_path
    config.kv_transfer_config.kv_connector = "SpyreExampleConnector"
    config.kv_transfer_config.kv_role = "kv_producer"
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


class TestSpyreExampleConnectorInit:
    """Tests for connector initialization."""

    def test_init_scheduler_role(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config()
        connector = SpyreExampleConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
        )
        assert connector._storage_path is not None
        assert connector._block_size == SPYRE_BLOCK_SIZE

    def test_init_worker_role(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config()
        connector = SpyreExampleConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        assert connector._storage_path is not None


class TestSpyreExampleConnectorSaveLoad:
    """Tests for KV cache save and load operations."""

    @pytest.fixture
    def tmp_storage(self):
        """Create a temporary directory for KV cache storage."""
        tmp_dir = tempfile.mkdtemp(prefix="spyre_kv_test_")
        yield tmp_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.fixture
    def worker_connector(self, tmp_storage):
        """Create a worker-side connector."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(storage_path=tmp_storage)
        connector = SpyreExampleConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states = _make_kv_states()
        connector.register_spyre_kv_caches(kv_states)
        return connector

    def test_save_kv_creates_files(self, worker_connector, tmp_storage):
        """Test that save_kv_layer creates safetensors files."""
        pytest.importorskip("safetensors")

        token_ids = list(range(64))
        meta = SpyreKVConnectorMetadata()
        meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=[1],
            is_store=True,
        )
        worker_connector.bind_connector_metadata(meta)

        # Save both layers
        for layer_idx in range(2):
            worker_connector.save_kv_layer(
                layer_name=f"layer_{layer_idx}",
                kv_layer=torch.empty(0),  # Not used for Spyre connector
                attn_metadata=MagicMock(),
            )

        # Check files were created
        entries = os.listdir(tmp_storage)
        assert len(entries) == 1  # One hash folder
        folder = os.path.join(tmp_storage, entries[0])
        assert os.path.exists(os.path.join(folder, "layer_0.safetensors"))
        assert os.path.exists(os.path.join(folder, "layer_1.safetensors"))

    def test_save_load_roundtrip(self, tmp_storage):
        """Test that saving and loading produces identical KV data."""
        safetensors = pytest.importorskip("safetensors")

        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        # Create save connector with known KV data
        config = _make_mock_vllm_config(storage_path=tmp_storage)
        save_connector = SpyreExampleConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states_original = _make_kv_states(num_layers=2)
        save_connector.register_spyre_kv_caches(kv_states_original)

        token_ids = list(range(64))
        block_ids = [1]

        # Save
        meta = SpyreKVConnectorMetadata()
        meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=block_ids,
            is_store=True,
        )
        save_connector.bind_connector_metadata(meta)

        for layer_idx in range(2):
            save_connector.save_kv_layer(
                layer_name=f"layer_{layer_idx}",
                kv_layer=torch.empty(0),
                attn_metadata=MagicMock(),
            )
        save_connector.clear_connector_metadata()

        # Create load connector with zeroed KV data
        load_connector = SpyreExampleConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states_loaded = _make_kv_states(num_layers=2)
        # Zero out the block we'll load into
        for layer in kv_states_loaded:
            layer[0][1].zero_()  # block 1 key
            layer[1][1].zero_()  # block 1 value
        load_connector.register_spyre_kv_caches(kv_states_loaded)

        # Load
        load_meta = SpyreKVConnectorMetadata()
        load_meta.add_new_request(
            req_id="req-1",
            token_ids=token_ids,
            block_ids=block_ids,
            is_store=False,
        )
        load_connector.bind_connector_metadata(load_meta)

        forward_ctx = MagicMock()
        load_connector.start_load_kv(forward_ctx)

        # Verify: the loaded KV data should match the original
        slot_mapping = SpyreKVConnectorBase.compute_slot_mapping(
            block_ids=block_ids, block_size=SPYRE_BLOCK_SIZE, num_tokens=64
        )
        for layer_idx in range(2):
            k_orig, v_orig = SpyreKVConnectorBase.extract_kv_for_slots(
                kv_states_original, layer_idx, slot_mapping
            )
            k_loaded, v_loaded = SpyreKVConnectorBase.extract_kv_for_slots(
                kv_states_loaded, layer_idx, slot_mapping
            )
            assert torch.allclose(k_orig, k_loaded), (
                f"Layer {layer_idx} key mismatch"
            )
            assert torch.allclose(v_orig, v_loaded), (
                f"Layer {layer_idx} value mismatch"
            )

    def test_wait_for_layer_load_is_noop(self, worker_connector):
        """wait_for_layer_load should be a no-op for sync connector."""
        worker_connector.wait_for_layer_load("layer_0")

    def test_wait_for_save_is_noop(self, worker_connector):
        """wait_for_save should be a no-op for sync connector."""
        worker_connector.wait_for_save()


class TestSpyreExampleConnectorScheduler:
    """Tests for scheduler-side connector methods."""

    @pytest.fixture
    def scheduler_connector(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        tmp_dir = tempfile.mkdtemp(prefix="spyre_kv_sched_")
        config = _make_mock_vllm_config(storage_path=tmp_dir)
        connector = SpyreExampleConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
        )
        yield connector
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_get_num_new_matched_tokens_no_cache(self, scheduler_connector):
        """Should return 0 when no cached KV exists."""
        request = MagicMock()
        request.prompt_token_ids = list(range(128))
        request.request_id = "req-1"

        num_matched, is_async = scheduler_connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )
        assert num_matched == 0
        assert is_async is False

    def test_build_connector_meta_store(self, scheduler_connector):
        """Should mark new requests as store when no cache exists."""
        scheduler_output = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "req-1"
        new_req.prompt_token_ids = list(range(128))
        new_req.block_ids = [[0, 1]]
        scheduler_output.scheduled_new_reqs = [new_req]

        meta = scheduler_connector.build_connector_meta(scheduler_output)
        assert isinstance(meta, SpyreKVConnectorMetadata)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True
        assert meta.requests[0].req_id == "req-1"

    def test_build_connector_meta_load(self, scheduler_connector):
        """Should mark requests needing load when in _requests_need_load."""
        request = MagicMock()
        request.request_id = "req-1"
        scheduler_connector._requests_need_load["req-1"] = request

        scheduler_output = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "req-1"
        new_req.prompt_token_ids = list(range(128))
        new_req.block_ids = [[0, 1]]
        scheduler_output.scheduled_new_reqs = [new_req]

        meta = scheduler_connector.build_connector_meta(scheduler_output)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is False

    def test_update_state_after_alloc(self, scheduler_connector):
        """Should track requests with external tokens."""
        request = MagicMock()
        request.request_id = "req-1"
        blocks = MagicMock()

        scheduler_connector.update_state_after_alloc(
            request, blocks, num_external_tokens=64
        )
        assert "req-1" in scheduler_connector._requests_need_load

    def test_update_state_after_alloc_no_external(self, scheduler_connector):
        """Should not track requests without external tokens."""
        request = MagicMock()
        request.request_id = "req-1"
        blocks = MagicMock()

        scheduler_connector.update_state_after_alloc(
            request, blocks, num_external_tokens=0
        )
        assert "req-1" not in scheduler_connector._requests_need_load
