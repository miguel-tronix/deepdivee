# Mozilla AI Libraries — Usage Review & Improvements

**any-agent**: `/home/migtronix/any-agent`
**any-llm**: `/home/migtronix/any-llm`
**DeepDive application**: `/home/migtronix/deepdivee`

---

## Overview

This document captures findings from a code review of how the DeepDive application uses the Mozilla AI libraries `any-agent` and `any-llm`. The review focused on `src/deepdive/agent/`.

---

## High Priority

### 1. `RedisSaver` checkpointer is dead code

**Files**: `src/deepdive/agent/memory.py:43`, `src/deepdive/agent/agent.py`

`AgentMemory` creates a `RedisSaver` instance (`self.checkpointer`) but never passes it to `AnyAgent`. The `AgentConfig` class in `any-agent` does not expose a direct `checkpointer` field — instead, the checkpointer must be passed via the `agent_args` dict:

```python
# agent.py — current
AgentConfig(
    model_id=f"{settings.llm_provider}:{settings.llm_model}",
    instructions=SYSTEM_PROMPT,
    tools=[],
)

# agent.py — fix
AgentConfig(
    model_id=f"{settings.llm_provider}:{settings.llm_model}",
    instructions=SYSTEM_PROMPT,
    tools=[],
    agent_args={"checkpointer": memory_store.checkpointer},
)
```

Without this, the LangGraph state persistence is inactive. The `RedisSaver` is instantiated but serves no purpose.

### 2. Hardcoded session ID — privacy data leak

**File**: `src/deepdive/agent/agent.py`

The conversation history is stored under a single, global session key:

```python
await memory_store.add_to_history("contraindications", {"role": "user", "content": prompt})
```

All users share the same Redis list for conversation history. The session ID must be per-user (e.g., derived from a user token, request ID, or authenticated user identifier):

```python
# Example: derive from request context
session_id = f"user:{user_id}:{intervention}"
await memory_store.add_to_history(session_id, ...)
```

---

## Medium Priority

### 3. `rag.py` `analyze_contraindications()` is dead code

**File**: `src/deepdive/agent/rag.py`

The file imports `AnyLLM`, creates a singleton `_llm`, and defines a full `analyze_contraindications()` async function — but **no route ever calls it**. The `/contraindications` endpoint in `routes.py` calls `analyze_with_agent()` from `agent.py` instead. The entire `any-llm` completion path in `rag.py` is unreachable dead code.

Options:
- **Remove** the dead code if `rag.py`'s `analyze_contraindications()` is not needed.
- **Wire it up** if the intent is to support multiple analysis paths (agent-based vs. direct LLM).

### 4. Global singletons not safe for concurrent initialization

**Files**: `src/deepdive/agent/agent.py`, `src/deepdive/agent/rag.py`, `src/deepdive/agent/embedders.py`

All three modules use module-level global singletons with lazy initialization:

```python
_agent: Optional[AnyAgent] = None

async def get_agent():
    global _agent
    if _agent is None:
        _agent = await AnyAgent.create_async(...)
    return _agent
```

If `get_agent()` is called concurrently from multiple async tasks before `_agent` is set, it can create multiple agent instances. Use an `asyncio.Lock` to guard initialization:

```python
_lock = asyncio.Lock()

async def get_agent():
    global _agent
    if _agent is None:
        async with _lock:
            if _agent is None:  # double-check under lock
                _agent = await AnyAgent.create_async(...)
    return _agent
```

Apply the same pattern to `_llm` in `rag.py` and `_embedder` in `embedders.py`.

### 5. No timeout on `acompletion()` calls

**File**: `src/deepdive/agent/rag.py`

The LLM completion call has no timeout configured:

```python
response = await llm.acompletion(
    model=settings.llm_model,
    messages=[...],
    temperature=0.2,
)
```

Without an explicit timeout, a slow or unresponsive LLM provider can cause requests to hang indefinitely. Add a timeout parameter:

```python
response = await llm.acompletion(
    model=settings.llm_model,
    messages=[...],
    temperature=0.2,
    timeout=60.0,  # seconds
)
```

### 6. Bare `except Exception` swallows exception details

**File**: `src/deepdive/agent/rag.py`

```python
except Exception as e:
    raise RuntimeError(f"Error in analyze_contraindications: {e}") from e
```

The original exception is chained via `from e`, but the exception type and stack trace context are lost. Consider:
- Raising a domain-specific exception (e.g., `ContraindicationAnalysisError`).
- Including the original traceback for debugging.
- Logging the error before re-raising.

---

## Low Priority

### 7. Using `AnyAgent` as a stateless LLM call

**File**: `src/deepdive/agent/agent.py`

`tools=[]` means the agent has no tool-calling capability. `AnyAgent` is being used as a thin LLM wrapper with the full agent overhead (framework initialization, tracing, etc.). Two options:

- **Use `any-llm` directly** (via the pattern in `rag.py`) since this is effectively a stateless completion call.
- **Add real tools** — RAG retrieval, web search, or other capabilities — to justify the agent abstraction.

If the intent is to keep the agent path but add capability later, this is fine to defer.

### 8. No TTL on conversation history

**File**: `src/deepdive/agent/memory.py`

`add_to_history()` appends to a Redis list with no expiration. Long-running sessions grow unbounded. Consider adding `EXPIRE` after a threshold, or using a sliding window:

```python
await self.client.expire(key, max_age_seconds=86400 * 30)
```

### 9. JSON serialization for Redis history

**File**: `src/deepdive/agent/memory.py`

Uses `json.dumps` / `json.loads` for serialization. For performance-sensitive paths, consider `orjson` (faster) or `msgpack` (binary, compact). This is low impact unless history operations are a bottleneck.

### 10. No Redis error handling

**File**: `src/deepdive/agent/memory.py`

All Redis operations raise unhandled exceptions if the connection is unavailable. Wrap operations with retry logic or graceful degradation:

```python
try:
    await self._client.get(key)
except redis.exceptions.ConnectionError:
    # Log warning, return empty/cached result
    return None
```

### 11. Hardcoded embedding dimension

**File**: `src/deepdive/db/models.py`

`Vector(768)` is hardcoded. The dimension should match the actual model being used. Pull from a config or settings rather than a literal:

```python
dimension=settings.embedding_dimension
```

---

## Summary Table

| Priority | Issue | File(s) |
|----------|-------|---------|
| High | `RedisSaver` checkpointer not passed to AnyAgent | `memory.py`, `agent.py` |
| High | Hardcoded session ID — all users share history | `agent.py`, `memory.py` |
| Medium | `rag.py` `analyze_contraindications()` is dead code | `rag.py` |
| Medium | Singleton race condition on concurrent init | `agent.py`, `rag.py`, `embedders.py` |
| Medium | No timeout on `acompletion()` calls | `rag.py` |
| Medium | `except Exception` loses exception details | `rag.py` |
| Low | `AnyAgent` used without tools (stateless LLM wrapper) | `agent.py` |
| Low | No TTL on conversation history | `memory.py` |
| Low | JSON serialization for Redis history | `memory.py` |
| Low | No Redis error handling | `memory.py` |
| Low | Hardcoded embedding dimension `768` | `models.py` |