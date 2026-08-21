import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.asyncio


def _fake_result(content: str) -> dict:
    return {"messages": [SimpleNamespace(content=content)]}


async def test_get_agent_creates_and_caches(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = MagicMock()
    monkeypatch.setattr(agent_module, "_agent", None)

    monkeypatch.setattr(agent_module.memory_store, "initialize", AsyncMock())
    monkeypatch.setattr(agent_module.memory_store, "checkpointer", MagicMock())

    monkeypatch.setattr(
        agent_module, "ChatOpenAI", MagicMock(return_value=MagicMock())
    )
    create_mock = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(agent_module, "create_react_agent", create_mock)

    result1 = await agent_module.get_agent()
    assert result1 is fake_agent
    create_mock.assert_called_once()

    result2 = await agent_module.get_agent()
    assert result2 is fake_agent
    create_mock.assert_called_once()


async def test_get_agent_wires_tools_checkpointer_and_instructions(monkeypatch):
    from deepdive.agent import agent as agent_module

    monkeypatch.setattr(agent_module, "_agent", None)
    fake_checkpointer = MagicMock()

    monkeypatch.setattr(agent_module.memory_store, "initialize", AsyncMock())
    monkeypatch.setattr(agent_module.memory_store, "checkpointer", fake_checkpointer)

    monkeypatch.setattr(
        agent_module, "ChatOpenAI", MagicMock(return_value=MagicMock())
    )
    create_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(agent_module, "create_react_agent", create_mock)

    await agent_module.get_agent()

    kwargs = create_mock.call_args.kwargs
    assert kwargs["tools"] == [agent_module.retrieve_pubmed_context]
    assert kwargs["checkpointer"] is fake_checkpointer
    assert kwargs["state_modifier"] == agent_module._INSTRUCTIONS


async def test_analyze_with_agent_runs_agent_and_stores_history(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = AsyncMock()
    fake_agent.ainvoke.return_value = _fake_result(
        "Contraindications: bleeding risk"
    )
    monkeypatch.setattr(agent_module, "_agent", fake_agent)

    fake_memory = AsyncMock()
    monkeypatch.setattr(agent_module, "memory_store", fake_memory)

    result = await agent_module.analyze_with_agent("aspirin")

    assert result == "Contraindications: bleeding risk"
    fake_agent.ainvoke.assert_awaited_once()
    args, kwargs = fake_agent.ainvoke.call_args
    state, config = args[0], kwargs["config"]
    assert "aspirin" in state["messages"][0][1]
    assert config["configurable"]["thread_id"] == "intervention:aspirin"

    assert fake_memory.add_to_history.await_count == 2
    fake_memory.add_to_history.assert_any_await(
        session_id="intervention:aspirin",
        role="user",
        content="Analyse contra-indications for: aspirin",
    )
    fake_memory.add_to_history.assert_any_await(
        session_id="intervention:aspirin",
        role="assistant",
        content="Contraindications: bleeding risk",
    )


async def test_analyze_with_agent_renders_prompt_with_intervention(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = AsyncMock()
    fake_agent.ainvoke.return_value = _fake_result("ok")
    monkeypatch.setattr(agent_module, "_agent", fake_agent)
    monkeypatch.setattr(agent_module, "memory_store", AsyncMock())

    await agent_module.analyze_with_agent("ibuprofen")

    state = fake_agent.ainvoke.call_args[0][0]
    assert "Intervention: ibuprofen" in state["messages"][0][1]


async def test_cleanup_agent_resets(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = MagicMock()
    monkeypatch.setattr(agent_module, "_agent", fake_agent)

    await agent_module.cleanup_agent()

    assert agent_module._agent is None


async def test_cleanup_agent_noop_when_none(monkeypatch):
    from deepdive.agent import agent as agent_module

    monkeypatch.setattr(agent_module, "_agent", None)

    await agent_module.cleanup_agent()

    assert agent_module._agent is None
