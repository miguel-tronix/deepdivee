"""
Redis-backed agent memory using LangGraph checkpointer.

Provides caching for analysis results and a LangGraph-compatible
checkpointer for agent state persistence across sessions.
"""

from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis
from langgraph.checkpoint.redis import RedisSaver

from deepdive.core.config import settings


class AgentMemory:
    """Redis-backed memory for agent state, caching, and conversation history."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: Optional[aioredis.Redis] = None
        self.checkpointer: Optional[RedisSaver] = None

    async def initialize(self) -> None:
        self._client = aioredis.from_url(self.redis_url, decode_responses=True)
        self.checkpointer = RedisSaver(async_client=self._client)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    @property
    def client(self) -> aioredis.Redis:
        assert self._client is not None, (
            "AgentMemory not initialised – call .initialize()"
        )
        return self._client

    # ------------------------------------------------------------------
    # Analysis-result caching
    # ------------------------------------------------------------------
    async def get_cached_analysis(self, intervention: str) -> Optional[str]:
        return await self.client.get(f"analysis:{intervention}")

    async def cache_analysis(
        self, intervention: str, analysis: str, ttl: int = 3600
    ) -> None:
        await self.client.setex(f"analysis:{intervention}", ttl, analysis)

    # ------------------------------------------------------------------
    # Conversation history (for future conversational features)
    # ------------------------------------------------------------------
    async def add_to_history(self, session_id: str, role: str, content: str) -> None:
        key = f"history:{session_id}"
        await self.client.rpush(key, json.dumps({"role": role, "content": content}))

    async def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        key = f"history:{session_id}"
        items = await self.client.lrange(key, -limit, -1)
        return [json.loads(item) for item in items]

    async def clear_history(self, session_id: str) -> None:
        await self.client.delete(f"history:{session_id}")


memory_store = AgentMemory(settings.redis_url)
