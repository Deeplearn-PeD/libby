#!/bin/bash
set -e

echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    sleep 1
done

# Pull the embedding model if specified
EMBEDDING_MODEL=${EMBEDDING_MODEL:-mxbai-embed-large}
echo "Pulling embedding model: $EMBEDDING_MODEL"
ollama pull "$EMBEDDING_MODEL" &

echo "Starting Libby API server..."
cd /app
exec uv run libby-server --host 0.0.0.0 --port 8000
