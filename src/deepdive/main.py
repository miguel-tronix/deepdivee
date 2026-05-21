from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from deepdive.api.exceptions import DeepDiveError
from deepdive.api.routes import router as api_router
from deepdive.core.config import settings
from deepdive.agent.embedders import initialise_embedder
from deepdive.agent.memory import memory_store
from deepdive.agent.agent import cleanup_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise embedder + Redis-backed memory at startup."""
    initialise_embedder()
    await memory_store.initialize()
    yield
    await memory_store.close()
    await cleanup_agent()


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="DeepDive RAG Agent - specialized in contra-indications",
    lifespan=lifespan,
)


@app.exception_handler(DeepDiveError)
async def deepdive_error_handler(request: Request, exc: DeepDiveError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.project_name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deepdive.main:app", host="0.0.0.0", port=8000, reload=True)
