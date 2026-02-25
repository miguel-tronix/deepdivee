from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from deepdive.core.config import settings

# Create async engine for PostgreSQL
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_size=5,
    max_overflow=10
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)

async def get_db():
    """Dependency for getting async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
