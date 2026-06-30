import pytest
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.asyncio


async def test_get_agent_creates_and_caches(monkeypatch):
    from deepdive.agent import agent as agent_module
    from deepdive.agent.agent import create_react_agent

    fake_agent = AsyncMock(spec=create_react_agent)
    monkeypatch.setattr(agent_module, "_agent", None)

    create_mock = AsyncMock(return_value=fake_agent)
    monkeypatch.setattr(agent_module, "create_react_agent", create_mock)

    monkeypatch.setattr(agent_module.memory_store, "initialize", AsyncMock())
    monkeypatch.setattr(agent_module.memory_store, "checkpointer", MagicMock())

    result1 = await agent_module.get_agent()
    assert result1 is fake_agent
    create_mock.assert_awaited_once()

    result2 = await agent_module.get_agent()
    assert result2 is fake_agent
    create_mock.assert_awaited_once()


async def test_analyze_with_agent_runs_agent_and_stores_history(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = AsyncMock()
    fake_result = MagicMock()
    fake_result.__getitem__ = lambda self, idx: self.messages[idx] if idx == -1 else None
    fake_result.messages = [MagicMock(content="Contraindications: bleeding risk")]
    fake_agent.ainvoke = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(agent_module, "_agent", fake_agent)

    fake_memory = AsyncMock()
    monkeypatch.setattr(agent_module, "memory_store", fake_memory)

    result = await agent_module.analyze_with_agent("aspirin")

    assert result == "Contraindications: bleeding risk"
    fake_agent.ainvoke.assert_awaited_once()
    call_args = fake_agent.ainvoke.call_args
    messages = call_args[0][0]["messages"]
    assert len(messages) == 1
    assert "aspirin" in messages[0].content

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


async def test_cleanup_agent_resets(monkeypatch):
    from deepdive.agent import agent as agent_module

    fake_agent = AsyncMock()
    monkeypatch.setattr(agent_module, "_agent", fake_agent)

    await agent_module.cleanup_agent()

    assert agent_module._agent is None


async def test_cleanup_agent_noop_when_none(monkeypatch):
    from deepdive.agent import agent as agent_module

    monkeypatch.setattr(agent_module, "_agent", None)

    await agent_module.cleanup_agent()

    assert agent_module._agent is None


async def test_get_agent_passes_correct_config(monkeypatch):
    from deepdive.agent import agent as agent_module

    monkeypatch.setattr(agent_module, "_agent", None)

    monkeypatch.setattr(agent_module.memory_store, "initialize", AsyncMock())
    monkeypatch.setattr(agent_module.memory_store, "checkpointer", MagicMock())

    fake_agent = AsyncMock()

    create_react_agent_mock = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(agent_module, "create_react_agent", create_react_agent_mock)

    await agent_module.get_agent()

    create_call_args = create_react_agent_mock.call_args
    tools_arg = create_call_args[1]["tools"]
    assert len(tools_arg) == 2
    assert "retrieve_pubmed_context" in [t.__name__ for t in tools_arg]
    assert "refine_search_query" in [t.__name__ for t in tools_arg]


async def test_refine_search_query_basic():
    from deepdive.agent import agent as agent_module

    result = await agent_module.refine_search_query(
        original_query="aspirin",
        refinement_reasoning="Need more specific information about dosage"
    )

    assert "aspirin" in result
    assert "dosage-related" in result
    assert " (dosage-related)" in result


async def test_refine_search_query_population_specific():
    from deepdive.agent import agent as agent_module

    result = await agent_module.refine_search_query(
        original_query="aspirin",
        refinement_reasoning="Looking for pediatric advice"
    )

    assert "aspirin" in result
    assert "pediatric" in result
    assert " (pediatric)" in result


async def test_refine_search_query_age_related():
    from deepdive.agent import agent as agent_module

    result = await agent_module.refine_search_query(
        original_query="aspirin",
        refinement_reasoning="Need information for elderly patients"
    )

    assert "aspirin" in result
    assert " (age-related)" in result


async def test_refine_search_query_multiple_indicators():
    from deepdive.agent import agent as agent_module

    result = await agent_module.refine_search_query(
        original_query="aspirin",
        refinement_reasoning="Looking for drug interactions and age-related information"
    )

    assert "aspirin" in result
    assert " (age-related)" in result
    assert " (drug interactions)" in result


async def test_refine_search_query_no_match():
    from deepdive.agent import agent as agent_module

    result = await agent_module.refine_search_query(
        original_query="aspirin",
        refinement_reasoning="Random text without matching indicators"
    )

    assert "aspirin" in result
    assert " (improved)" in result
