# Libby D. Bot

[![PyPI version](https://img.shields.io/pypi/v/libby.svg)](https://pypi.org/project/libby/)
[![Python](https://img.shields.io/pypi/pyversions/libby.svg)](https://pypi.org/project/libby/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub stars](https://img.shields.io/github/stars/Deeplearn-PeD/libby.svg?style=flat&logo=github&color=yellow)](https://github.com/Deeplearn-PeD/libby/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Deeplearn-PeD/libby.svg?style=flat&logo=github)](https://github.com/Deeplearn-PeD/libby/network/members)
[![GitHub issues](https://img.shields.io/github/issues/Deeplearn-PeD/libby.svg)](https://github.com/Deeplearn-PeD/libby/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Deeplearn-PeD/libby.svg)](https://github.com/Deeplearn-PeD/libby/pulls)
[![GitHub last commit](https://img.shields.io/github/last-commit/Deeplearn-PeD/libby.svg)](https://github.com/Deeplearn-PeD/libby/commits/main)
[![DOI](https://zenodo.org/badge/784398327.svg)](https://zenodo.org/doi/10.5281/zenodo.12744747)

Libby the librarian. AI agent specialized in creating and querying embeddings for RAG (Retrieval Augmented Generation). Now featuring an **LLM Wiki** — a persistent, compounding markdown knowledge base — and a rich **Terminal User Interface (TUI)**.

![Libby D. Bot](/libby.jpeg)

## What's New

- **Textual TUI** (`libby`) — A rich interactive terminal interface with screens for chat, embedding, wiki browsing, and settings.
- **LLM Wiki** (`libby-cli wiki_*`) — A persistent markdown knowledge base that accumulates insights from your documents over time, with entity/concept extraction, synthesis, and automated health-checks.
- **Legacy CLI** (`libby-cli`) — The original Fire-based command-line interface preserved for scripting.

## Installation

You can install Libby D. Bot using pip:

```bash
pip install -U libby
```

Or using uv:

```bash
uv sync
```

## Usage

### Interactive TUI (Recommended)

Launch the rich terminal interface:

```bash
uv run libby
```

The TUI opens on the **Dashboard** showing your collections and quick actions. Use keyboard shortcuts to navigate:

| Shortcut | Screen |
|----------|--------|
| `Ctrl+D` | Dashboard |
| `Ctrl+C` | Chat |
| `Ctrl+E` | Embed Documents |
| `Ctrl+W` | Wiki Browser |
| `Ctrl+S` | Settings |
| `Ctrl+Q` | Quit |

#### Chat Screen

Ask questions about your documents in an interactive chat. Switch between modes:
- **RAG Answer** — Retrieve relevant chunks and answer
- **Free Generate** — Creative generation with document context
- **Wiki Query** — Query the accumulated LLM Wiki

Chat history is persisted to the database and loaded on each session.

#### Embed Screen

Browse the filesystem, select a folder of PDFs, and embed them into a collection with live progress logging.

#### Wiki Browser

Navigate your collection's markdown wiki as a tree:
- `sources/` — One page per ingested document
- `entities/` — Extracted people, organizations, objects
- `concepts/` — Topics, theories, ideas
- `synthesis/` — Overviews, analyses, answers

Select any page to render it as Markdown. Use **Refresh** to reload, **Ingest** to add documents, or **Lint** to health-check the wiki.

#### Settings Screen

Change the active LLM model, embedding model, and collection name.

### Legacy CLI (Scripting)

For automation and scripts, use the Fire-based CLI:

```bash
# Creating embeddings
libby-cli embed --corpus_path /path/to/your/documents --collection_name your_collection

# Querying documents
libby-cli answer "What is the main topic of the documents?" --collection_name your_collection

# Generating content
libby-cli generate "Write a summary of..." --output_file output.txt
```

## LLM Wiki

The LLM Wiki is a persistent, compounding markdown knowledge base that sits between your raw documents and your questions. Instead of re-deriving knowledge from scratch on every query, Libby **incrementally builds and maintains a wiki** — extracting entities, concepts, and synthesis that grow richer with every document you add.

### Wiki Directory Structure

Each collection gets its own wiki under `~/.libby/wikis/<collection_name>/`:

```
<collection_name>/
├── index.md          # Content-oriented catalog of all pages
├── log.md            # Chronological append-only record of operations
├── sources/          # One page per ingested source document
├── entities/         # Pages for people, organizations, objects
├── concepts/         # Pages for topics, theories, ideas
└── synthesis/        # Overviews, answers, comparisons, analyses
```

All pages use YAML frontmatter and Obsidian-style `[[wikilinks]]`:

```markdown
---
title: Page Title
date_created: 2026-04-21T10:00:00
---

# Page Title

Content here with [[Other Page]] links.
```

### Wiki Commands

```bash
# Ingest documents into the wiki
libby-cli wiki_ingest --corpus_path /path/to/docs --collection_name my_collection

# Query the wiki
libby-cli wiki_query "What is the main topic?" --collection_name my_collection --file_answer

# Health-check the wiki
libby-cli wiki_lint --collection_name my_collection --auto_fix

# Show wiki statistics
libby-cli wiki_status --collection_name my_collection
```

### Wiki Workflows

**Ingest** — When you add a source, Libby:
1. Generates a structured summary (entities, concepts, contradictions)
2. Plans which wiki pages to create/update
3. Writes/updates source, entity, concept, and synthesis pages
4. Updates `index.md` and appends to `log.md`

**Query** — When you ask a question:
1. Reads `index.md` to identify relevant pages
2. Reads the most relevant pages (up to 15)
3. Synthesizes a cited answer using `[[Page Name]]` citations
4. Optionally files the answer back into `synthesis/`

**Lint** — Periodic health-checks scan for:
- **Orphan pages** — pages with zero inbound wikilinks
- **Broken links** — wikilinks pointing to non-existent pages
- **Contradictions** — conflicting claims between pages
- **Stale claims** — claims superseded by newer sources
- **Missing pages** — important terms lacking dedicated pages

Auto-fix creates stub pages for broken links with `status: stub` frontmatter.

## REST API Server

Libby D. Bot provides a REST API server for programmatic access to embedding, retrieval, and wiki functionality.

### Running the Server

**Using the CLI:**

```bash
# Run with default settings (host: 0.0.0.0, port: 8000)
uv run libby-server

# Run with custom host and port
uv run libby-server --host 0.0.0.0 --port 8080

# Run with auto-reload for development
uv run libby-server --reload
```

**Using uvicorn directly:**

```bash
uv run uvicorn libbydbot.api.main:app --host 0.0.0.0 --port 8000
```

**Using Docker:**

```bash
# Copy and configure environment variables
cp .env.example .env
# IMPORTANT: Edit .env and set a secure POSTGRES_PASSWORD

# Build and run with Docker Compose (recommended)
# This will start PostgreSQL with pgvector, the API, and backup services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

> **Note:**
> - Port 8001 is used to avoid conflicts. Change to `8000:8000` if port 8000 is available.
> - The Docker setup includes:
>   - **PostgreSQL with pgvector**: Default database backend with vector similarity search
>   - **Ollama server**: With the `mxbai-embed-large` embedding model pre-installed
>   - **Automatic backups**: Daily backups at 2 AM (configurable)
>   - **Wiki persistence**: Wikis are stored in the `libby-wikis` volume mounted at `/wikis`
> - Models are persisted in the `ollama-models` volume for faster restarts.
> - PostgreSQL data is persisted in the `postgres-data` volume.
> - **Security**: You MUST set a secure `POSTGRES_PASSWORD` in your `.env` file before running.

### API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/embed/text` | Embed raw text into the database |
| `POST` | `/api/embed/upload` | Upload and embed PDF files |
| `POST` | `/api/retrieve` | Hybrid search for documents (vector + keyword) |
| `GET`  | `/api/documents` | List all embedded documents |
| `GET`  | `/api/collections` | List all collections with document counts |
| `POST` | `/api/wiki/ingest` | Ingest a source into the wiki |
| `POST` | `/api/wiki/query` | Query the wiki |
| `POST` | `/api/wiki/lint` | Lint the wiki |
| `GET`  | `/api/wiki/status/{collection_name}` | Wiki statistics |
| `GET`  | `/api/health` | Health check endpoint |

### API Usage Examples

#### POST /api/embed/text

Embed raw text content into the vector database.

**Request:**

```bash
curl -X POST "http://localhost:8001/api/embed/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of artificial intelligence...",
    "doc_name": "ml_notes.txt",
    "page_number": 0,
    "collection_name": "research"
  }'
```

**Response:**

```json
{
  "success": true,
  "doc_name": "ml_notes.txt",
  "doc_hash": "abc123def456",
  "message": "Successfully embedded text from 'ml_notes.txt' into collection 'research'"
}
```

#### POST /api/wiki/ingest

Ingest a document into the LLM Wiki.

**Request:**

```bash
curl -X POST "http://localhost:8001/api/wiki/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "research_paper.pdf",
    "doc_content": "The full text of the document...",
    "collection_name": "research",
    "source_type": "document"
  }'
```

**Response:**

```json
{
  "success": true,
  "source": "research_paper.pdf",
  "pages_touched": 5,
  "entities_created": 3,
  "concepts_created": 2,
  "summary": "This paper explores...",
  "message": "Successfully ingested 'research_paper.pdf' into wiki 'research'"
}
```

#### POST /api/wiki/query

Query the wiki and synthesize an answer.

**Request:**

```bash
curl -X POST "http://localhost:8001/api/wiki/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "collection_name": "research",
    "file_answer": false
  }'
```

**Response:**

```json
{
  "question": "What are the key findings?",
  "answer": "The key findings include... [[Source Page]]",
  "sources_used": ["sources/research_paper.md", "concepts/key_finding.md"],
  "confidence": "high",
  "gaps": [],
  "suggested_followups": ["What methodology was used?"],
  "filed_page": null
}
```

#### POST /api/wiki/lint

Health-check the wiki.

**Request:**

```bash
curl -X POST "http://localhost:8001/api/wiki/lint" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "research",
    "auto_fix": true
  }'
```

**Response:**

```json
{
  "orphan_pages": ["entities/old_entity.md"],
  "broken_links": ["MissingConcept"],
  "contradictions": [],
  "stale_claims": [],
  "missing_pages": [{"name": "MissingConcept", "page_type": "concept"}],
  "suggestions": ["Add more cross-references between sources."],
  "fixes_applied": 1
}
```

#### GET /api/wiki/status/{collection_name}

Get wiki statistics.

**Request:**

```bash
curl "http://localhost:8001/api/wiki/status/research"
```

**Response:**

```json
{
  "collection": "research",
  "wiki_path": "/home/user/.libby/wikis/research",
  "total_pages": 42,
  "page_counts": {"sources": 10, "entities": 15, "concepts": 12, "synthesis": 5},
  "orphan_pages": 2,
  "broken_links": 1,
  "last_operation": "## [2026-04-21 12:00] ingest | research_paper.pdf | touched 5 pages"
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_DB` | PostgreSQL database name | `libby` |
| `POSTGRES_USER` | PostgreSQL username | `libby` |
| `POSTGRES_PASSWORD` | PostgreSQL password (REQUIRED) | - |
| `EMBED_DB` | Database URL for embeddings (auto-generated) | `postgresql://libby:***@postgres:5432/libby` |
| `EMBEDDING_MODEL` | Embedding model to use | `mxbai-embed-large` |
| `COLLECTION_NAME` | Default collection name | `main` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `WIKI_BASE_PATH` | Base directory for LLM wikis | `~/.libby/wikis` |
| `WIKI_AUTO_INGEST` | Auto-ingest into wiki after embedding | `False` |
| `BACKUP_RETENTION_DAYS` | Days to keep PostgreSQL backups | `7` |
| `BACKUP_SCHEDULE` | Backup schedule (cron format) | `0 2 * * *` |

## Database Configuration

### PostgreSQL (Default/Recommended)

Libby D. Bot now uses PostgreSQL with the pgvector extension as the default database backend. This provides:

- **Better scalability** for large document collections
- **Production-ready** performance and reliability
- **Vector similarity search** using pgvector extension
- **Automatic backups** with configurable retention

### Migrating from DuckDB

**📋 Complete Migration Guide**: See [docs/COMPLETE_MIGRATION_GUIDE.md](docs/COMPLETE_MIGRATION_GUIDE.md) for comprehensive instructions.

#### Quick Migration

If you have existing embeddings in DuckDB, you can easily migrate them to PostgreSQL:

```bash
# Ensure PostgreSQL is running
docker compose up -d postgres

# Simple migration (if dimensions match)
./scripts/migrate.sh

# Migration with re-embedding (if dimensions don't match)
./scripts/migrate.sh --re-embed
```

#### What Gets Migrated

- ✅ Document embeddings
- ✅ Document metadata (names, page numbers, hashes)
- ✅ **Collection names** (preserved from DuckDB)
- ✅ Document text content

#### Migration Features

- **Automatic dimension detection** - Detects 768 vs 1024 dimension mismatches
- **Re-embedding support** - Automatically re-embeds when needed
- **Collection preservation** - Maintains original collection names
- **Progress tracking** - Real-time progress with time estimation
- **Dry-run mode** - Preview without making changes
- **Resume capability** - Continue interrupted migrations

For detailed instructions, troubleshooting, and advanced options, see the [Complete Migration Guide](docs/COMPLETE_MIGRATION_GUIDE.md).

### Alternative Database Backends

While PostgreSQL is recommended, Libby also supports:

- **DuckDB**: Good for development/testing (`duckdb:///path/to/embeddings.duckdb`)
- **SQLite**: For simple use cases (`sqlite:///path/to/embeddings.db`)

To use an alternative backend, set the `EMBED_DB` environment variable accordingly.

## Backup and Restore

### Automated Backups

The Docker setup includes automated PostgreSQL backups:

- **Schedule**: Daily at 2 AM (configurable via `BACKUP_SCHEDULE`)
- **Retention**: 7 days (configurable via `BACKUP_RETENTION_DAYS`)
- **Location**: Stored in the `postgres-backups` volume

### Manual Backup

```bash
# Create a manual backup
docker compose exec postgres-backup /backup.sh --manual

# List available backups
docker compose exec postgres-backup /backup.sh --list
```

### Restore from Backup

```bash
# Copy backup from container
docker cp libby-postgres-backup:/backups/libby_backup_YYYYMMDD_HHMMSS.sql.gz ./

# Restore to PostgreSQL
gunzip -c libby_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker compose exec -T postgres psql -U libby -d libby
```

## Features

- **Rich Textual TUI** — Interactive terminal interface with dashboards, chat, embedding, and wiki browsing
- **LLM Wiki** — Persistent markdown knowledge base with entity/concept extraction, synthesis, and health-checks
- **Multiple language support** (English and Portuguese)
- **Various AI models** available (Llama3, Gemma, ChatGPT, Qwen, Gemini)
- **PDF document processing** and embedding with chunking
- **Question answering** with context from your documents
- **Content generation** capabilities
- **REST API** for programmatic access
- **Docker support** for containerized deployment
- **Obsidian-compatible** wiki format

## Configuration

Libby supports different AI models and languages. You can configure these through environment variables or the config.yml file.

Available Models:
- Llama3 (default)
- Gemma
- ChatGPT
- Qwen
- Gemini

Supported Languages:
- English (en_US)
- Portuguese (pt_BR)

## License

This project is licensed under the GPLv3 License.
