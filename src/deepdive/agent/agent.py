"""
AnyAgent wrapper for contra-indication analysis.

Uses `any-agent` with the LangChain framework to create a portable
agent that leverages `any-llm` under the hood and supports
Redis-backed memory via LangGraph checkpointers.
"""

from __future__ import annotations

from typing import Optional

from any_agent import AgentConfig, AnyAgent

from deepdive.core.config import settings
from deepdive.agent.memory import memory_store

_agent: Optional[AnyAgent] = None


async def get_agent() -> AnyAgent:
    global _agent
    if _agent is None:
        _agent = await AnyAgent.create_async(
            "langchain",
            AgentConfig(
                model_id=f"{settings.llm_provider}:{settings.llm_model}",
                instructions=(
                    "You are an expert medical AI specialising in identifying "
                    "contra-indications. Analyse interventions based strictly "
                    "on the provided PubMed literature context."
                ),
                tools=[],
            ),
        )
    return _agent


async def analyze_with_agent(intervention: str, context: str) -> str:
    prompt = f"""
Based strictly on the provided PubMed literature context, analyse the following
intervention and list its major contra-indications.

Intervention: {intervention}

Context:
{context}

Response format: Provide a concise summary followed by bullet points of
contra-indications.
"""
    agent = await get_agent()
    result = await agent.run_async(prompt)

    await memory_store.add_to_history(
        session_id="contraindications",
        role="user",
        content=f"Analyse contra-indications for: {intervention}",
    )
    await memory_store.add_to_history(
        session_id="contraindications",
        role="assistant",
        content=result.final_output,
    )

    return result.final_output


async def cleanup_agent() -> None:
    global _agent
    if _agent is not None:
        await _agent.cleanup_async()
        _agent = None
