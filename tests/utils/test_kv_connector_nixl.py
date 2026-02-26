"""Tests for NixlConnector (network-based P/D disaggregation)."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_spyre.v1.kv_connector.base import (
    SPYRE_BLOCK_SIZE,
    align_to_spyre_block,
)
from vllm_spyre.v1.kv_connector.nixl_connector import (
    NixlReqMeta,
    NixlConnector,
    NixlConnectorMetadata,
    NixlHandshakeMetadata,
)


def _make_mock_vllm_config(kv_role="kv_producer"):
    """Create a mock VllmConfig for testing."""
    config = MagicMock()
    config.cache_config.block_size = SPYRE_BLOCK_SIZE
    config.kv_transfer_config.kv_connector = "NixlConnector"
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_rank = 0
    config.kv_transfer_config.kv_parallel_size = 2
    config.kv_transfer_config.engine_id = "test-engine-0"
    config.kv_transfer_config.get_from_extra_config.return_value = None
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


class TestNixlConnectorInit:
    """Tests for connector initialization."""

    def test_init_producer(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(kv_role="kv_producer")
        connector = NixlConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        assert connector._is_producer is True
        assert connector._is_consumer is False
        assert connector._engine_id == "test-engine-0"

    def test_init_consumer(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(kv_role="kv_consumer")
        connector = NixlConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        assert connector._is_producer is False
        assert connector._is_consumer is True


class TestNixlHandshakeMetadata:
    """Tests for handshake metadata."""

    def test_create_metadata(self):
        meta = NixlHandshakeMetadata(
            agent_name="test_agent",
            agent_metadata=b"test_data",
            engine_id="engine-0",
        )
        assert meta.agent_name == "test_agent"
        assert meta.engine_id == "engine-0"

    def test_get_handshake_metadata_without_nixl(self):
        """Should return None when NIXL is not available."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config()
        connector = NixlConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        # NIXL agent may or may not be initialized depending on environment
        metadata = connector.get_handshake_metadata()
        # Either None (no nixl) or NixlHandshakeMetadata
        assert metadata is None or isinstance(
            metadata, NixlHandshakeMetadata
        )


class TestNixlConnectorMetadata:
    """Tests for NIXL connector metadata."""

    def test_empty_metadata(self):
        meta = NixlConnectorMetadata()
        assert len(meta.requests) == 0

    def test_nixl_req_meta(self):
        req = NixlReqMeta(
            req_id="req-1",
            token_ids=list(range(64)),
            num_tokens=64,
            block_ids=[0],
            is_store=True,
            remote_engine_id="engine-1",
        )
        assert req.remote_engine_id == "engine-1"


class TestNixlConnectorProducer:
    """Tests for producer-side operations."""

    @pytest.fixture
    def producer(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(kv_role="kv_producer")
        conn = NixlConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states = _make_kv_states()
        conn.register_spyre_kv_caches(kv_states)
        return conn

    def test_save_kv_layer_marks_ready(self, producer):
        """Producer should track layers that are ready for remote reads."""
        meta = NixlConnectorMetadata()
        meta.requests.append(
            NixlReqMeta(
                req_id="req-1",
                token_ids=list(range(64)),
                num_tokens=64,
                block_ids=[1],
                is_store=True,
            )
        )
        producer.bind_connector_metadata(meta)

        producer.save_kv_layer(
            layer_name="layer_0",
            kv_layer=torch.empty(0),
            attn_metadata=MagicMock(),
        )

        assert "req-1" in producer._pending_transfers
        assert "layer_0" in producer._pending_transfers["req-1"]

    def test_wait_for_save_noop(self, producer):
        """wait_for_save should be a no-op for producer."""
        producer.wait_for_save()

    def test_start_load_kv_noop_for_producer(self, producer):
        """start_load_kv should be a no-op for producer."""
        meta = NixlConnectorMetadata()
        producer.bind_connector_metadata(meta)
        producer.start_load_kv(MagicMock())  # Should not raise


class TestNixlConnectorConsumer:
    """Tests for consumer-side operations."""

    @pytest.fixture
    def consumer(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(kv_role="kv_consumer")
        conn = NixlConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
        )
        kv_states = _make_kv_states()
        conn.register_spyre_kv_caches(kv_states)
        return conn

    def test_start_load_kv_handles_empty_metadata(self, consumer):
        """Should handle empty metadata gracefully."""
        meta = NixlConnectorMetadata()
        consumer.bind_connector_metadata(meta)
        consumer.start_load_kv(MagicMock())  # Should not raise

    def test_wait_for_layer_load_noop(self, consumer):
        """wait_for_layer_load should be a no-op."""
        consumer.wait_for_layer_load("layer_0")


class TestNixlConnectorScheduler:
    """Tests for scheduler-side methods."""

    @pytest.fixture
    def scheduler_producer(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(kv_role="kv_producer")
        return NixlConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
        )

    @pytest.fixture
    def scheduler_consumer(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        config = _make_mock_vllm_config(kv_role="kv_consumer")
        return NixlConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
        )

    def test_producer_returns_zero_matched(self, scheduler_producer):
        """Producer should always return 0 matched tokens."""
        request = MagicMock()
        request.prompt_token_ids = list(range(128))
        request.request_id = "req-1"

        num_matched, is_async = scheduler_producer.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )
        assert num_matched == 0
        assert is_async is False

    def test_consumer_returns_zero_without_remote_kv(self, scheduler_consumer):
        """Consumer should return 0 when no remote KV is available."""
        request = MagicMock()
        request.prompt_token_ids = list(range(128))
        request.request_id = "req-1"

        num_matched, is_async = scheduler_consumer.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )
        assert num_matched == 0
        assert is_async is False

    def test_build_connector_meta_producer(self, scheduler_producer):
        """Producer should mark new requests as store."""
        scheduler_output = MagicMock()
        new_req = MagicMock()
        new_req.req_id = "req-1"
        new_req.prompt_token_ids = list(range(128))
        new_req.block_ids = [[0, 1]]
        scheduler_output.scheduled_new_reqs = [new_req]

        meta = scheduler_producer.build_connector_meta(scheduler_output)
        assert isinstance(meta, NixlConnectorMetadata)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True

    def test_shutdown(self, scheduler_producer):
        """Shutdown should clean up state."""
        scheduler_producer.shutdown()
        assert scheduler_producer._nixl_agent is None
        assert len(scheduler_producer._peer_agents) == 0
