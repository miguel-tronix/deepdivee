# Use a slim python base image
FROM python:3.12-slim

# Install astral-sh/uv to dramatically speed up installations
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the dependency requirement files first (to cache the dependency layer independent of code changes)
COPY pyproject.toml uv.lock ./

# Install python dependencies without installing the project code yet
RUN uv sync --frozen --no-cache --no-install-project

# Now copy the rest of the application
COPY src/ /app/src/
COPY docs/ /app/docs/
COPY README.md /app/

# Sync again to install the project
RUN uv sync --frozen --no-cache

# Set up the execution environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# The port our FastAPI application will run on
EXPOSE 8000

# Start the uvicorn server
CMD ["uvicorn", "deepdive.main:app", "--host", "0.0.0.0", "--port", "8000"]
