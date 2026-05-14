import json

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_redis():
    client = AsyncMock()
    return client


async def test_init_stores_redis_url():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    assert mem.redis_url == "redis://test:6379/0"
    assert mem._client is None
    assert mem.checkpointer is None


async def test_initialize_creates_client_and_checkpointer(monkeypatch):
    from deepdive.agent import memory as mem_module
    from deepdive.agent.memory import AgentMemory, RedisSaver

    mem = AgentMemory("redis://test:6379/0")

    fake_client = AsyncMock()
    fake_checkpointer = MagicMock(spec=RedisSaver)

    monkeypatch.setattr(mem_module.aioredis, "from_url", lambda url, decode_responses: fake_client)

    mock_redis_saver = MagicMock()
    monkeypatch.setattr(mem_module, "RedisSaver", lambda async_client: fake_checkpointer)

    await mem.initialize()

    assert mem._client is fake_client
    assert mem.checkpointer is fake_checkpointer


async def test_close_closes_client():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    mem._client = fake_client

    await mem.close()

    fake_client.close.assert_awaited_once()


async def test_close_noop_when_no_client():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    await mem.close()


async def test_client_property_raises_when_uninitialized():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    with pytest.raises(AssertionError, match="not initialised"):
        _ = mem.client


async def test_client_property_returns_client_when_initialized():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    mem._client = fake_client

    assert mem.client is fake_client


async def test_get_cached_analysis_returns_none_when_missing():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    fake_client.get.return_value = None
    mem._client = fake_client

    result = await mem.get_cached_analysis("aspirin")
    assert result is None
    fake_client.get.assert_awaited_once_with("analysis:aspirin")


async def test_get_cached_analysis_returns_value():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    fake_client.get.return_value = "some analysis"
    mem._client = fake_client

    result = await mem.get_cached_analysis("ibuprofen")
    assert result == "some analysis"


async def test_cache_analysis_sets_with_ttl():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    mem._client = fake_client

    await mem.cache_analysis("paracetamol", "safe", ttl=7200)
    fake_client.setex.assert_awaited_once_with("analysis:paracetamol", 7200, "safe")


async def test_cache_analysis_defaults_ttl_to_3600():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    mem._client = fake_client

    await mem.cache_analysis("aspirin", "bleeding risk")
    fake_client.setex.assert_awaited_once_with("analysis:aspirin", 3600, "bleeding risk")


async def test_add_to_history_pushes_json():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    mem._client = fake_client

    await mem.add_to_history("session_1", "user", "hello")

    expected = json.dumps({"role": "user", "content": "hello"})
    fake_client.rpush.assert_awaited_once_with("history:session_1", expected)


async def test_get_history_returns_parsed_items():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    items = [
        json.dumps({"role": "user", "content": "q1"}),
        json.dumps({"role": "assistant", "content": "a1"}),
    ]
    fake_client.lrange.return_value = items
    mem._client = fake_client

    result = await mem.get_history("session_1", limit=10)

    assert result == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    fake_client.lrange.assert_awaited_once_with("history:session_1", -10, -1)


async def test_get_history_defaults_limit_to_20():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    fake_client.lrange.return_value = []
    mem._client = fake_client

    await mem.get_history("session_1")
    fake_client.lrange.assert_awaited_once_with("history:session_1", -20, -1)


async def test_get_history_handles_empty():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    fake_client.lrange.return_value = []
    mem._client = fake_client

    result = await mem.get_history("empty_session")
    assert result == []


async def test_clear_history_deletes_key():
    from deepdive.agent.memory import AgentMemory

    mem = AgentMemory("redis://test:6379/0")
    fake_client = AsyncMock()
    mem._client = fake_client

    await mem.clear_history("session_1")
    fake_client.delete.assert_awaited_once_with("history:session_1")


async def test_memory_store_is_agentmemory_instance(monkeypatch):
    from deepdive.agent.memory import AgentMemory, memory_store

    assert isinstance(memory_store, AgentMemory)
