#!/bin/bash
set -e

echo "Checking Ollama service availability..."
OLLAMA_HOST=${OLLAMA_HOST:-http://ollama:11434}

for i in {1..30}; do
    if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "Ollama service is ready at $OLLAMA_HOST!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Warning: Ollama service not responding after 30 seconds"
        echo "Continuing startup, but embedding operations may fail"
    fi
    sleep 1
done

# Pull the embedding model if specified
EMBEDDING_MODEL=${EMBEDDING_MODEL:-mxbai-embed-large}
echo "Ensuring embedding model is available: $EMBEDDING_MODEL"
curl -s -X POST "$OLLAMA_HOST/api/pull" -d "{\"name\": \"$EMBEDDING_MODEL\"}" > /dev/null 2>&1 || \
    echo "Warning: Could not pull embedding model. It may already exist or Ollama is unavailable."

echo "Starting Libby API server..."
cd /app
exec uv run libby-server --host 0.0.0.0 --port 8000
