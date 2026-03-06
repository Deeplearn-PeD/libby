FROM python:3.12-slim

LABEL maintainer="Flávio Codeço Coelho <fccoelho@gmail.com>"
LABEL description="Libby D. Bot API - AI-powered librarian for RAG document embedding and retrieval"

# Install system dependencies for PyMuPDF (PDF processing) and Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

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

# Copy startup script
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV EMBED_DB=duckdb:///data/embeddings.duckdb
ENV OLLAMA_HOST=http://localhost:11434

# Expose ports (8000 for API, 11434 for Ollama)
EXPOSE 8000 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the startup script
CMD ["/start.sh"]
