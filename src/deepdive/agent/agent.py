"""
AnyAgent wrapper for contra-indication analysis.

Uses LangGraph and LangChain to create a portable
agent that leverages OpenAI-compatible models under the hood, uses a PubMed vector-search
tool to retrieve relevant literature, and supports Redis-backed memory
via LangGraph checkpointers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from deepdive.core.config import settings
from deepdive.agent.memory import memory_store
from deepdive.agent.rag import retrieve_pubmed_context
from deepdive.agent.templating import render

_INSTRUCTIONS = render("system_prompt.jinja2")

_agent: Any = None
_agent_lock = asyncio.Lock()


async def get_agent() -> Any:
    global _agent
    if _agent is None:
        async with _agent_lock:
            if _agent is None:
                await memory_store.initialize()

                from pydantic import SecretStr
                model = ChatOpenAI(
                    model=settings.llm_model,
                    api_key=SecretStr(settings.llm_api_key),
                    base_url=settings.llm_api_base,
                )

                _agent = create_react_agent(
                    model=model,
                    tools=[retrieve_pubmed_context],
                    checkpointer=memory_store.checkpointer,
                    state_modifier=_INSTRUCTIONS,
                )
    return _agent


async def analyze_with_agent(intervention: str) -> str:
    prompt = render("analysis_prompt.jinja2", intervention=intervention)
    agent = await get_agent()

    session_id = f"intervention:{intervention}"
    config = {"configurable": {"thread_id": session_id}}

    result = await agent.ainvoke({"messages": [("user", prompt)]}, config=config)
    output_str = result["messages"][-1].content

    await memory_store.add_to_history(
        session_id=session_id,
        role="user",
        content=f"Analyse contra-indications for: {intervention}",
    )
    await memory_store.add_to_history(
        session_id=session_id,
        role="assistant",
        content=output_str,
    )

    return output_str


async def cleanup_agent() -> None:
    global _agent
    _agent = None
