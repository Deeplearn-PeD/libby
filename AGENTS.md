# AGENTS.md - Coding Agent Guidelines for Libby D. Bot

Libby D. Bot is an AI-powered librarian for RAG. It processes PDF documents, generates embeddings, and enables question answering over document collections.

## Build/Lint/Test Commands

### Package Manager (uv)

```bash
uv sync                    # Install dependencies
uv add <package-name>      # Add a dependency
uv add --dev <package-name> # Add a dev dependency
```

### Running Tests

```bash
uv run pytest                           # Run all tests
uv run pytest tests/test_cli.py         # Run a single test file
uv run pytest tests/test_cli.py::TestLibbyInterface::test_initialization_default  # Single test
uv run pytest -v                        # Verbose output
uv run pytest tests/brain/              # Run tests in directory
uv run pytest -k "embed"                # Run tests matching pattern
```

### Running the CLI

```bash
uv run libby embed --corpus_path /path/to/docs --collection_name my_collection
uv run libby answer "What is the main topic?" --collection_name my_collection
uv run libby generate "Write a summary..." --output_file output.txt
```

## Code Style Guidelines

### Imports

Order: standard library → third-party → local. Separate groups with blank lines.

```python
# Standard library
import os
from glob import glob

# Third-party
import fitz
import loguru
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field, create_engine, Session

# Local
from libbydbot.brain.ingest import PDFPipeline
from libbydbot.settings import Settings
```

### Type Annotations

Use Python 3.10+ union syntax (`int | None` instead of `Optional[int]`). Use `list[str]` and `dict[str, Any]` instead of `List[str]` and `Dict[str, Any]`.

### Naming Conventions

- **Classes**: PascalCase (`DocEmbedder`, `PDFPipeline`, `LibbyInterface`)
- **Functions/Methods**: snake_case (`embed_text`, `retrieve_docs`)
- **Private methods**: Prefix with underscore (`_generate_embedding`)
- **Constants**: UPPER_SNAKE_CASE (`PROVIDERS`)
- **Module singletons**: lowercase (`logger`, `settings`)

### Docstrings

```python
def embed_text(self, doctext: str, docname: str, page_number: int):
    """
    Embed a page of a document.
    :param doctext: page of a document
    :param docname: name of the document
    :param page_number: page number
    """
```

### Error Handling

Use try/except with loguru logging. Log levels: `logger.info()` for normal operations, `logger.warning()` for recoverable issues, `logger.error()` for errors.

```python
try:
    session.add(doc_vector)
    session.commit()
except IntegrityError as e:
    session.rollback()
    logger.warning(f"Document {docname} already exists: {e}")
except ValueError as e:
    logger.error(f"Error: {e} generated when processing: {doctext}")
    session.rollback()
```

### Class Structure

Order: `__init__` → `@property` → public methods → private methods (`_` prefix) → dunder methods.

### Database Patterns

Supports SQLite, DuckDB, and PostgreSQL:

```python
if self.dburl.startswith("sqlite"):
    # SQLite-specific code
elif self.dburl.startswith("duckdb"):
    # DuckDB-specific code
else:
    # PostgreSQL code
```

### Configuration

Use pydantic-settings with `SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")`.

### Testing

Both unittest.TestCase and pytest styles are acceptable. Use pytest fixtures for mocking:

```python
@pytest.fixture(autouse=True)
def mock_embeddings():
    with patch('libbydbot.brain.embed.DocEmbedder._generate_embedding') as mocked:
        mocked.return_value = np.zeros(1024).tolist()
        yield mocked
```

### Async Patterns

Use `nest_asyncio.apply()` and `asyncio.run()` for async operations.

## Project Structure

```
libby/
├── libbydbot/           # Main package
│   ├── __init__.py      # Persona class
│   ├── cli.py           # CLI interface (Fire-based)
│   ├── settings.py      # Pydantic settings
│   └── brain/           # Core functionality
│       ├── __init__.py  # LibbyDBot class
│       ├── embed.py     # Document embedding
│       ├── ingest.py    # PDF ingestion
│       ├── memory.py    # Chat history
│       └── analyze.py   # Article summarization
├── tests/
│   ├── conftest.py      # Shared fixtures
│   └── brain/
└── pyproject.toml
```

## Environment Variables

- `PGURL` - PostgreSQL connection URL
- `OLLAMA_HOST` - Ollama server URL (default: http://localhost:11434)
- `GEMINI_API_KEY` - Google Gemini API key

## Supported Models

Llama3.2 (default), Gemma3, GPT-4o, Qwen3, Gemini

## Database URLs

- SQLite: `sqlite:///path/to/db.sqlite`
- DuckDB: `duckdb:///path/to/db.duckdb`
- PostgreSQL: Full postgresql:// URL
