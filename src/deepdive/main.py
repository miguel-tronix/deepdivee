from fastapi import FastAPI
from deepdive.api.routes import router as api_router
from deepdive.core.config import settings

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="DeepDive RAG Agent - specialized in contra-indications",
)

app.include_router(api_router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.project_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deepdive.main:app", host="0.0.0.0", port=8000, reload=True)
