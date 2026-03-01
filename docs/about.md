# About Libby D. Bot

## Overview

Libby D. Bot is an AI-powered librarian designed for Retrieval Augmented Generation (RAG). It processes PDF documents, generates vector embeddings, and enables intelligent question answering over document collections.

## Architecture

```
libbydbot/
├── __init__.py          # Persona class
├── cli.py               # CLI interface (Fire-based)
├── settings.py          # Pydantic settings configuration
├── persona_prompts.py   # Persona prompt templates
└── brain/               # Core functionality
    ├── __init__.py      # LibbyDBot class
    ├── embed.py         # Document embedding & retrieval
    ├── ingest.py        # PDF ingestion & chunking
    ├── memory.py        # Chat history persistence
    └── analyze.py       # Article summarization
```

### Key Components

#### 1. LibbyDBot (`brain/__init__.py`)

The main agent class that orchestrates:
- LLM integration via `base-ai-agent`
- Document retrieval tool registration
- Context management for RAG

#### 2. DocEmbedder (`brain/embed.py`)

Handles:
- Vector embedding generation
- Multi-database support (SQLite, DuckDB, PostgreSQL)
- Hybrid search (vector + keyword/FTS)
- Document chunking and storage

#### 3. PDFPipeline (`brain/ingest.py`)

Manages:
- PDF text extraction using PyMuPDF
- Text chunking with configurable overlap
- Metadata extraction

#### 4. History (`brain/memory.py`)

Persists:
- Conversation history
- User-message associations
- Context tracking

#### 5. ArticleSummarizer (`brain/analyze.py`)

Provides:
- Structured article summarization
- Key information extraction (research question, results, conclusions)

## Features

### Multi-Database Support

Libby supports three database backends:

| Database | URL Format | Features |
|----------|------------|----------|
| SQLite | `sqlite:///path/to/db.sqlite` | Lightweight, sqlite-vec for vectors |
| DuckDB | `duckdb:///path/to/db.duckdb` | Fast analytics, VSS extension |
| PostgreSQL | `postgresql://...` | Production-ready, pgvector |

### Hybrid Search

Libby combines two search methods:

1. **Vector Search**: Semantic similarity using embeddings
2. **Keyword Search**: Full-text search (FTS5/FTS)

Results are merged using Reciprocal Rank Fusion (RRF) for optimal relevance.

### Supported LLM Models

| Model | Code | Provider |
|-------|------|----------|
| Llama3 | `llama3.2` | Ollama |
| Gemma | `gemma3` | Google/Ollama |
| GPT-4o | `gpt-4o` | OpenAI |
| Qwen | `qwen3` | Ollama |
| Gemini | `gemini` | Google |

### Supported Embedding Models

| Model | Code | Dimensions | Provider |
|-------|------|------------|----------|
| GemmaEmbedding | `embeddinggemma` | 1024 | Ollama |
| Mxbai | `mxbai-embed-large` | 1024 | Ollama |
| Gemini | `gemini-embedding-001` | 1024 | Google |

### REST API

FastAPI-based REST API with:
- CORS middleware
- Interactive documentation (Swagger/ReDoc)
- Health check endpoint
- Docker-ready deployment

## Configuration

### Settings (`settings.py`)

Configuration is managed via pydantic-settings:

```python
class Settings(BaseSettings):
    languages: Dict[str, Dict[str, Any]] = {
        "English": {"code": "en_US", "is_default": True},
        "Português": {"code": "pt_BR"},
    }
    
    models: Dict[str, Dict[str, Any]] = {
        "Llama3": {"code": "llama3.2", "is_default": True},
        "Gemma": {"code": "gemma3"},
        "ChatGPT": {"code": "gpt-4o"},
        "Qwen": {"code": "qwen3"},
    }
    
    embedding_models: Dict[str, Dict[str, Any]] = {
        "GemmaEmbedding": {"code": "embeddinggemma", "is_default": True},
        "Mxbai": {"code": "mxbai-embed-large"},
        "Gemini": {"code": "gemini-embedding-001"},
    }
```

### Environment Variables

Load from `.env` file automatically:

```bash
OLLAMA_HOST=http://localhost:11434
GEMINI_API_KEY=your_key
PGURL=postgresql://user:pass@host/db
```

## Dependencies

Key dependencies include:
- **PyMuPDF** (`fitz`): PDF processing
- **SQLAlchemy/SQLModel**: Database ORM
- **pgvector/sqlite-vec/DuckDB VSS**: Vector storage
- **Ollama**: Local LLM inference
- **FastAPI/Uvicorn**: REST API
- **pydantic-settings**: Configuration management
- **base-ai-agent**: LLM abstraction layer

## Project Structure

```
libby/
├── libbydbot/           # Main package
│   ├── brain/           # Core functionality
│   └── api/             # REST API
├── tests/               # Test suite
├── docs/                # Documentation
├── data/                # Default data directory
├── pyproject.toml       # Project configuration
├── mkdocs.yml           # Documentation config
├── Dockerfile           # Docker configuration
└── docker-compose.yml   # Docker Compose setup
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0-only).
