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
│       ├── analyze.py   # Article summarization
│       ├── wiki.py      # LLM Wiki manager
│       └── wiki_models.py # Structured wiki schemas
├── tests/
│   ├── conftest.py      # Shared fixtures
│   └── brain/
└── pyproject.toml
```

## Environment Variables

- `PGURL` - PostgreSQL connection URL
- `OLLAMA_HOST` - Ollama server URL (default: http://localhost:11434)
- `GEMINI_API_KEY` - Google Gemini API key
- `WIKI_BASE_PATH` - Base directory for LLM wikis (default: `~/.libby/wikis`)
- `WIKI_AUTO_INGEST` - Automatically ingest documents into wiki after embedding (default: `False`)

## Supported Models

Llama3.2 (default), Gemma3, GPT-4o, Qwen3, Gemini

## Database URLs

- SQLite: `sqlite:///path/to/db.sqlite`
- DuckDB: `duckdb:///path/to/db.duckdb`
- PostgreSQL: Full postgresql:// URL

## Release Process

Use the release script to commit changes, bump version, and create git tags:

```bash
./scripts/release.sh
```

The script will:
1. Show current git status and pending changes
2. Ask for version bump type (major/minor/patch)
3. Update version in `pyproject.toml`
4. Commit all changes with a message
5. Create an annotated git tag (e.g., `v0.9.0`)

After running the script, push changes with:
```bash
git push && git push --tags
```

### Version Bump Guidelines

- **patch**: Bug fixes, documentation updates, small tweaks (0.8.0 → 0.8.1)
- **minor**: New features, backward-compatible changes (0.8.0 → 0.9.0)
- **major**: Breaking changes, major refactors (0.8.0 → 1.0.0)

## LLM Wiki Conventions

Libby maintains an **LLM Wiki** — a persistent, compounding markdown knowledge base that sits between raw embedded documents and the user. The wiki is stored as Obsidian-compatible markdown files.

### Wiki Directory Structure

Each collection gets its own wiki under `WIKI_BASE_PATH` (default: `~/.libby/wikis/<collection_name>/`):

```
<collection_name>/
├── index.md          # Content-oriented catalog of all pages
├── log.md            # Chronological append-only record of operations
├── sources/          # One page per ingested source document
├── entities/         # Pages for people, organizations, objects
├── concepts/         # Pages for topics, theories, ideas
└── synthesis/        # Overviews, answers, comparisons, analyses
```

### Page Format

All pages use YAML frontmatter and Obsidian-style `[[wikilinks]]`:

```markdown
---
title: Page Title
date_created: 2026-04-21T10:00:00
---

# Page Title

Content here with [[Other Page]] links.
```

### Ingest Workflow

1. Read source content (from raw PDF or embedded chunks).
2. Generate `SourceSummary` (entities, concepts, contradictions).
3. Generate `WikiUpdatePlan` (pages to create/update).
4. Write/update:
   - `sources/<doc_name>.md`
   - `entities/<entity>.md` (create or append mention)
   - `concepts/<concept>.md` (create or append mention)
   - `synthesis/overview.md` (if synthesis notes exist)
5. Update `index.md` and append to `log.md`.

### Query Workflow

1. Read `index.md` to identify candidate pages.
2. Read the most relevant pages (up to 15, keyword-ranked).
3. LLM synthesizes a cited answer using `[[Page Name]]` citations.
4. Optionally file the answer back into `synthesis/` and update `index.md`.

### Lint Workflow

Periodic health-checks scan for:
- **Orphan pages** — pages with zero inbound wikilinks
- **Broken links** — wikilinks pointing to non-existent pages
- **Contradictions** — conflicting claims between pages
- **Stale claims** — claims superseded by newer sources
- **Missing pages** — important terms mentioned but lacking dedicated pages

Auto-fix creates stub pages for broken links with `status: stub` frontmatter.

### CLI Commands

```bash
uv run libby wiki_ingest --corpus_path /path/to/docs --collection_name my_collection
uv run libby wiki_query "What is the main topic?" --collection_name my_collection --file_answer
uv run libby wiki_lint --collection_name my_collection --auto_fix
uv run libby wiki_status --collection_name my_collection
```

### REST API Endpoints

- `POST /api/wiki/ingest` — ingest a source into the wiki
- `POST /api/wiki/query` — query the wiki
- `POST /api/wiki/lint` — lint the wiki
- `GET /api/wiki/status/{collection_name}` — wiki statistics
