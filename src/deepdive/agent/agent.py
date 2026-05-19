"""
AnyAgent wrapper for contra-indication analysis.

Uses `any-agent` with the LangChain framework to create a portable
agent that leverages `any-llm` under the hood, uses a PubMed vector-search
tool to retrieve relevant literature, and supports Redis-backed memory
via LangGraph checkpointers.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from any_agent import AgentConfig, AnyAgent

from deepdive.core.config import settings
from deepdive.agent.memory import memory_store
from deepdive.agent.rag import retrieve_pubmed_context

_INSTRUCTIONS = """You are an expert medical AI specialising in identifying contra-indications.

When asked to analyse a medical intervention, follow these steps:

1. Generate a precise, detailed search query using medical terminology.
   Make the query more robust than the original question — include
   related terms, drug classes, and condition synonyms.
2. Call the `retrieve_pubmed_context` tool with your improved query
   to get relevant PubMed literature.
3. Based strictly on the retrieved context, list the major
   contra-indications with PMID citations where available.
4. If the retrieved context is empty or insufficient, try a different
   search query to find more relevant results before giving up.
5. If no relevant literature can be found after multiple attempts,
   state that no PubMed evidence was found — do not make up or infer
   contra-indications without literature support.

Response format: Provide a concise summary followed by bullet points
of contra-indications."""

_agent: Optional[AnyAgent] = None
_agent_lock = asyncio.Lock()


async def get_agent() -> AnyAgent:
    global _agent
    if _agent is None:
        async with _agent_lock:
            if _agent is None:
                await memory_store.initialize()
                _agent = await AnyAgent.create_async(
                    "langchain",
                    AgentConfig(
                        model_id=f"{settings.llm_provider}:{settings.llm_model}",
                        instructions=_INSTRUCTIONS,
                        tools=[retrieve_pubmed_context],
                        agent_args={"checkpointer": memory_store.checkpointer},
                    ),
                )
    return _agent


async def analyze_with_agent(intervention: str) -> str:
    prompt = f"""Analyse the following intervention and list its major contra-indications.

Intervention: {intervention}

First, generate a detailed search query and call retrieve_pubmed_context
to get the relevant literature. Then provide your analysis based strictly
on the retrieved context."""
    agent = await get_agent()
    result = await agent.run_async(prompt)

    session_id = f"intervention:{intervention}"
    await memory_store.add_to_history(
        session_id=session_id,
        role="user",
        content=f"Analyse contra-indications for: {intervention}",
    )
    await memory_store.add_to_history(
        session_id=session_id,
        role="assistant",
        content=result.final_output,
    )

    return result.final_output


async def cleanup_agent() -> None:
    global _agent
    if _agent is not None:
        await _agent.cleanup_async()
        _agent = None
