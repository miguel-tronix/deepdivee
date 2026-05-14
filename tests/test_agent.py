import pytest
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.asyncio


async def test_get_agent_creates_and_caches(monkeypatch):
    from deepdive.agent import agent as agent_module
    from deepdive.agent.agent import AnyAgent

    fake_agent = AsyncMock(spec=AnyAgent)
    monkeypatch.setattr(agent_module, "_agent", None)

    create_mock = AsyncMock(return_value=fake_agent)
    monkeypatch.setattr(agent_module.AnyAgent, "create_async", create_mock)

    result1 = await agent_module.get_agent()
    assert result1 is fake_agent
    create_mock.assert_awaited_once()

    result2 = await agent_module.get_agent()
    assert result2 is fake_agent
    create_mock.assert_awaited_once()


async def test_analyze_with_agent_runs_agent_and_stores_history(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = AsyncMock()
    fake_run_result = MagicMock()
    fake_run_result.final_output = "Contraindications: bleeding risk"
    fake_agent.run_async = AsyncMock(return_value=fake_run_result)
    monkeypatch.setattr(agent_module, "_agent", fake_agent)

    fake_memory = AsyncMock()
    monkeypatch.setattr(agent_module, "memory_store", fake_memory)

    result = await agent_module.analyze_with_agent(
        "aspirin", "PubMed context about aspirin"
    )

    assert result == "Contraindications: bleeding risk"
    fake_agent.run_async.assert_awaited_once()
    assert "aspirin" in fake_agent.run_async.call_args[0][0]

    assert fake_memory.add_to_history.await_count == 2
    fake_memory.add_to_history.assert_any_await(
        session_id="contraindications",
        role="user",
        content="Analyse contra-indications for: aspirin",
    )
    fake_memory.add_to_history.assert_any_await(
        session_id="contraindications",
        role="assistant",
        content="Contraindications: bleeding risk",
    )


async def test_cleanup_agent_resets(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = AsyncMock()
    monkeypatch.setattr(agent_module, "_agent", fake_agent)

    await agent_module.cleanup_agent()

    fake_agent.cleanup_async.assert_awaited_once()
    assert agent_module._agent is None


async def test_cleanup_agent_noop_when_none(monkeypatch):
    from deepdive.agent import agent as agent_module

    monkeypatch.setattr(agent_module, "_agent", None)

    await agent_module.cleanup_agent()

    assert agent_module._agent is None


async def test_get_agent_passes_correct_config(monkeypatch):
    from deepdive.agent import agent as agent_module

    monkeypatch.setattr(agent_module, "_agent", None)

    fake_agent = AsyncMock()
    captured_config = {}

    async def fake_create(framework, config):
        captured_config["framework"] = framework
        captured_config["model_id"] = config.model_id
        captured_config["instructions"] = config.instructions
        return fake_agent

    monkeypatch.setattr(agent_module.AnyAgent, "create_async", fake_create)

    await agent_module.get_agent()

    assert captured_config["framework"] == "langchain"
    assert "openai" in captured_config["model_id"]
    assert "contra-indications" in captured_config["instructions"]
