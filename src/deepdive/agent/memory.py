"""
Redis-backed agent memory using LangGraph checkpointer.

Provides caching for analysis results and a LangGraph-compatible
checkpointer for agent state persistence across sessions.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import redis.asyncio as aioredis
from langgraph.checkpoint.redis import AsyncRedisSaver

from deepdive.core.config import settings

logger = logging.getLogger(__name__)


class AgentMemory:
    """Redis-backed memory for agent state, caching, and conversation history."""

    HISTORY_TTL = 86400 * 30  # 30 days

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: Optional[aioredis.Redis] = None
        self.checkpointer: Optional[AsyncRedisSaver] = None

    async def initialize(self) -> None:
        self._client = aioredis.from_url(self.redis_url, decode_responses=True)
        self.checkpointer = AsyncRedisSaver(redis_client=self._client)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    @property
    def client(self) -> aioredis.Redis:
        assert self._client is not None, (
            "AgentMemory not initialised – call .initialize()"
        )
        return self._client

    async def _redis_call(self, method: str, *args, **kwargs):
        """Call a redis method with error handling."""
        try:
            return await getattr(self.client, method)(*args, **kwargs)
        except aioredis.RedisError:
            logger.warning("Redis operation '%s' failed", method, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Analysis-result caching
    # ------------------------------------------------------------------
    async def get_cached_analysis(self, intervention: str) -> Optional[str]:
        result = await self._redis_call("get", f"analysis:{intervention}")
        return result

    async def cache_analysis(
        self, intervention: str, analysis: str, ttl: int = 3600
    ) -> None:
        await self._redis_call("setex", f"analysis:{intervention}", ttl, analysis)

    # ------------------------------------------------------------------
    # Conversation history (for future conversational features)
    # ------------------------------------------------------------------
    async def add_to_history(self, session_id: str, role: str, content: str) -> None:
        key = f"history:{session_id}"
        await self._redis_call(
            "rpush", key, json.dumps({"role": role, "content": content})
        )
        await self._redis_call("expire", key, self.HISTORY_TTL)

    async def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        key = f"history:{session_id}"
        items = await self._redis_call("lrange", key, -limit, -1)
        if items is None:
            return []
        return [json.loads(item) for item in items]

    async def clear_history(self, session_id: str) -> None:
        await self._redis_call("delete", f"history:{session_id}")


memory_store = AgentMemory(settings.redis_url)
