# Libby D. Bot

[![DOI](https://zenodo.org/badge/784398327.svg)](https://zenodo.org/doi/10.5281/zenodo.12744747)

Libby the librarian. AI agent specialized in creating and querying embeddings for RAG (Retrieval Augmented Generation).
![Libby D. Bot](/libby.jpeg)

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

Libby provides several commands through its CLI interface:

### Creating Embeddings

Create embeddings from your documents in a specified directory:

```bash
libby embed --corpus_path /path/to/your/documents --collection_name your_collection
```

The `corpus_path` defaults to the current directory if not specified. The `collection_name` parameter allows you to organize your embeddings into different collections (defaults to 'main').

### Querying Documents

After creating embeddings, you can ask questions about your documents:

```bash
libby answer "What is the main topic of the documents?" --collection_name your_collection
```

### Generating Content

You can use Libby to generate content based on prompts:

```bash
# Generate using direct prompt
libby generate "Write a summary of..." --output_file output.txt

# Generate using prompt from file
libby generate "" --prompt_file input_prompt.txt --output_file output.txt
```

## REST API Server

Libby D. Bot provides a REST API server for programmatic access to embedding and retrieval functionality.

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
| `GET` | `/api/documents` | List all embedded documents |
| `GET` | `/api/collections` | List all collections with document counts |
| `GET` | `/api/health` | Health check endpoint |

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

#### POST /api/embed/upload

Upload and embed a PDF file. The file is automatically chunked and embedded.

**Request:**

```bash
curl -X POST "http://localhost:8001/api/embed/upload" \
  -F "file=@document.pdf" \
  -F "collection_name=research" \
  -F "chunk_size=800" \
  -F "chunk_overlap=100"
```

**Response:**

```json
{
  "success": true,
  "doc_name": "document.pdf",
  "chunks_embedded": 15,
  "collection_name": "research",
  "message": "Successfully embedded 15 chunks from 'document.pdf' into collection 'research'"
}
```

#### POST /api/retrieve

Perform hybrid search (vector + keyword) across your documents.

**Request:**

```bash
curl -X POST "http://localhost:8001/api/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "collection_name": "research",
    "num_docs": 5
  }'
```

**Response:**

```json
{
  "query": "What is machine learning?",
  "collection_name": "research",
  "documents": [
    {
      "doc_name": "ml_intro.pdf",
      "page_number": 1,
      "content": "Machine learning is a subset of artificial intelligence...",
      "score": 0.95
    },
    {
      "doc_name": "ai_basics.pdf",
      "page_number": 5,
      "content": "ML algorithms learn patterns from data...",
      "score": 0.87
    }
  ],
  "total": 2
}
```

#### GET /api/documents

List all embedded documents, optionally filtered by collection.

**Request (all documents):**

```bash
curl "http://localhost:8001/api/documents"
```

**Request (filtered by collection):**

```bash
curl "http://localhost:8001/api/documents?collection_name=research"
```

**Response:**

```json
{
  "documents": [
    {"doc_name": "ml_intro.pdf", "collection_name": "research"},
    {"doc_name": "ai_basics.pdf", "collection_name": "research"},
    {"doc_name": "notes.txt", "collection_name": "personal"}
  ],
  "total": 3
}
```

#### GET /api/collections

List all collections with their document counts.

**Request:**

```bash
curl "http://localhost:8001/api/collections"
```

**Response:**

```json
{
  "collections": [
    {"name": "research", "document_count": 15},
    {"name": "personal", "document_count": 3},
    {"name": "work", "document_count": 7}
  ],
  "total": 3
}
```

#### GET /api/health

Check the API server health status.

**Request:**

```bash
curl "http://localhost:8001/api/health"
```

**Response:**

```json
{
  "status": "healthy",
  "database": "postgresql",
  "version": "0.1.0"
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

If you have existing embeddings in DuckDB, you can migrate them to PostgreSQL:

```bash
# Ensure PostgreSQL is running
docker compose up -d postgres

# Run the migration script
uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path /path/to/embeddings.duckdb \
  --postgres-url postgresql://libby:yourpassword@localhost:5432/libby
```

The migration script will:
- Create necessary tables in PostgreSQL
- Enable the pgvector extension
- Migrate all embeddings while preserving document hashes
- Skip duplicates automatically
- Show progress with a progress bar

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

- Multiple language support (English and Portuguese)
- Various AI models available (Llama3, Gemma, ChatGPT)
- PDF document processing and embedding
- Question answering with context from your documents
- Content generation capabilities
- REST API for programmatic access
- Docker support for containerized deployment

## Configuration

Libby supports different AI models and languages. You can configure these through environment variables or the config.yml file.

Available Models:
- Llama3 (default)
- Gemma
- Llama3-vision
- ChatGPT

Supported Languages:
- English (en_US)
- Portuguese (pt_BR)

## License

This project is licensed under the GPLv3 License.
