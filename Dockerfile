FROM python:3.12-slim

LABEL maintainer="Flávio Codeço Coelho <fccoelho@gmail.com>"
LABEL description="Libby D. Bot API - AI-powered librarian for RAG document embedding and retrieval"

# Install system dependencies for PyMuPDF (PDF processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY libbydbot ./libbydbot

# Install dependencies (no dev dependencies for production)
RUN uv sync --frozen --no-dev

# Create directory for persistent database files
RUN mkdir -p /data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV EMBED_DB=/data/embeddings.duckdb
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the API server
CMD ["uv", "run", "uvicorn", "libbydbot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
