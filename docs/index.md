# Libby D. Bot Documentation

Welcome to the official documentation for Libby D. Bot, your AI-powered librarian for RAG (Retrieval Augmented Generation).

## What is Libby D. Bot?

Libby D. Bot is an AI-powered librarian that:
- Processes PDF documents and generates vector embeddings
- Enables semantic search over document collections
- Provides question answering using retrieved context
- Builds a persistent, compounding **LLM Wiki** of entities, concepts, and synthesis
- Maintains a **knowledge graph** over wiki pages and embedded chunks, with path/explain queries and an interactive visualization
- Offers content generation capabilities
- Exposes a REST API for programmatic access

## Documentation Chapters

1. [Tutorial](tutorial.md) - Step-by-step guide to using Libby
2. [API Reference](reference.md) - Technical documentation of classes and methods
3. [About](about.md) - Project architecture and features
4. [Contact](contact.md) - How to get in touch
5. [License](license.md) - Licensing information

## Quick Start

### Installation

```bash
# Using pip
pip install -U libby

# Using uv (recommended)
uv sync
```

### CLI Usage

```bash
# Embed documents from a directory
libby embed --corpus_path ./documents

# Ask questions about your documents
libby answer "What is the main topic?"

# Generate content based on retrieved context
libby generate "Write a summary..." --output_file output.txt
```

### REST API

```bash
# Start the API server
uv run libby-server

# Access interactive docs at http://localhost:8000/docs
```

### Docker with SFTP

```bash
# Setup SSH keys
./scripts/setup-ssh-keys.sh

# Start all services (API + SFTP + Watcher)
docker-compose up -d

# Upload documents via SFTP
sftp -i ssh_keys/ssh_host_ed25519_key -P 2222 libby@localhost
sftp> put document.pdf
```

## Features

- **Multiple Database Support**: SQLite, DuckDB, and PostgreSQL
- **Hybrid Search**: Combines vector similarity with keyword search (FTS)
- **LLM Wiki**: Persistent markdown knowledge base with entity/concept extraction and health-checks
- **Knowledge Graph**: NetworkX graph over wiki pages and embedded chunks; graph-aware queries, path/explain navigation, and an interactive `graph.html` served by the REST API
- **Multiple LLM Models**: Llama3, Gemma, GPT-4o, Qwen3, Gemini
- **Embedding Models**: embeddinggemma (default), mxbai-embed-large, gemini-embedding-001
- **Multi-language**: English and Portuguese support
- **Docker Ready**: Containerized deployment support
- **SFTP Ingestion**: Automated document upload and processing via SFTP
- **Document Watcher**: Cron-based monitoring for new documents
