# Tutorial

This tutorial will guide you through using Libby D. Bot to create embeddings from your documents, build a persistent knowledge wiki, and query them interactively.

## Table of Contents

1. [Installation](#step-1-installation)
2. [Environment Setup](#step-2-setting-up-environment-variables)
3. [Preparing Documents](#step-3-preparing-your-documents)
4. [Using the TUI (Recommended)](#step-4-using-the-tui)
5. [Creating Embeddings](#step-5-creating-embeddings)
6. [Building the LLM Wiki](#step-6-building-the-llm-wiki)
7. [Querying Your Knowledge](#step-7-querying-your-documents)
8. [Using the REST API](#step-8-using-the-rest-api)
9. [Advanced Usage](#advanced-usage)

---

## Step 1: Installation

Install Libby using pip or uv:

```bash
# Using pip
pip install -U libby

# Using uv (recommended)
uv sync
```

## Step 2: Setting Up Environment Variables

Create a `.env` file in your project directory:

```bash
# Optional: Ollama server URL (default: http://localhost:11434)
OLLAMA_HOST=http://localhost:11434

# Optional: Gemini API key (for Gemini models)
GEMINI_API_KEY=your_api_key

# Optional: OpenAI API key (for GPT models)
OPENAI_API_KEY=your_api_key

# Optional: Database URL for embeddings
EMBED_DB=duckdb:///data/embeddings.duckdb

# Optional: Wiki base path (default: ~/.libby/wikis)
WIKI_BASE_PATH=~/.libby/wikis
```

## Step 3: Preparing Your Documents

Place your PDF documents in a directory:

```bash
mkdir -p documents
# Copy your PDFs to the documents folder
```

## Step 4: Using the TUI

Libby now features a rich **Textual TUI** (Terminal User Interface). Launch it with:

```bash
uv run libby
```

The TUI opens on the **Dashboard** showing your collections and quick actions.

### Keyboard Shortcuts

| Shortcut | Screen | Description |
|----------|--------|-------------|
| `Ctrl+D` | Dashboard | Collections overview, quick actions |
| `Ctrl+C` | Chat | Interactive Q&A with your documents |
| `Ctrl+E` | Embed | Select and embed documents with progress |
| `Ctrl+W` | Wiki Browser | Browse your markdown wiki |
| `Ctrl+S` | Settings | Change model, collection, config |
| `Ctrl+Q` | — | Quit the application |

### Dashboard

The Dashboard shows:
- A **Collections** table with document counts and embedding models
- **Quick Actions** buttons to jump to Chat, Embed, Wiki, or Settings

Select a collection in the table to activate it.

### Chat Screen

The Chat screen is an interactive conversation interface. Type your question and press Enter.

**Chat Modes:**
- **RAG Answer** — Retrieves relevant document chunks and answers
- **Free Generate** — Creative generation using document context
- **Wiki Query** — Queries the accumulated LLM Wiki (not raw documents)

Your chat history is persisted and loaded automatically on each session.

### Embed Screen

1. Browse the filesystem tree on the left
2. Select a folder containing PDFs
3. Set the collection name
4. Click **Start Embedding**

Progress appears in the log panel below, showing each document and chunk processed.

### Wiki Browser

Navigate your collection's markdown wiki as a hierarchical tree:
- `index.md` — Catalog of all pages
- `log.md` — Chronological history of operations
- `sources/` — One page per ingested document
- `entities/` — Extracted people, organizations, objects
- `concepts/` — Topics, theories, ideas
- `synthesis/` — Analyses, overviews, answers

Click any page to view it as rendered Markdown. Use **Refresh** to reload the tree, **Ingest** to add new sources, or **Lint** to run health-checks.

---

## Step 5: Creating Embeddings

### Using the TUI

Navigate to the **Embed** screen (`Ctrl+E`), select a folder, and click **Start Embedding**.

### Using the Legacy CLI

For scripting and automation:

```bash
# Basic usage
libby-cli embed --corpus_path ./documents

# With custom collection name
libby-cli embed --corpus_path ./documents --collection_name research
```

The embedding process:
1. Extracts text from PDFs using PyMuPDF
2. Splits text into chunks (default: 800 chars with 100 char overlap)
3. Generates vector embeddings using the default model (embeddinggemma)
4. Stores embeddings in the database

### Using Python

```python
from libbydbot.brain.embed import DocEmbedder

embedder = DocEmbedder(
    col_name="research",
    dburl="duckdb:///embeddings.duckdb",
    embedding_model="embeddinggemma"
)

# Embed all PDFs in a directory
embedder.embed_path("./documents")
```

---

## Step 6: Building the LLM Wiki

The **LLM Wiki** is Libby's most powerful feature: a persistent, compounding markdown knowledge base. Instead of retrieving raw chunks on every query, Libby reads your documents, extracts structured knowledge, and maintains interlinked markdown pages that grow richer over time.

### What the Wiki Contains

Each collection gets its own wiki under `~/.libby/wikis/<collection_name>/`:

```
research/
├── index.md          # Catalog of all pages
├── log.md            # Chronological operation log
├── sources/          # Summaries of each document
│   ├── paper_1.md
│   └── paper_2.md
├── entities/         # People, organizations, objects
│   ├── alice_smith.md
│   └── acme_corp.md
├── concepts/         # Topics, theories, ideas
│   ├── machine_learning.md
│   └── neural_networks.md
└── synthesis/        # Analyses and overviews
    └── overview.md
```

### Ingesting Documents into the Wiki

#### Using the TUI

1. Go to **Wiki Browser** (`Ctrl+W`)
2. Click **Ingest** (or press `i`)
3. Select a folder of PDFs
4. Click **Start Ingest**

You'll see progress for each document: pages touched, entities created, concepts created.

#### Using the Legacy CLI

```bash
libby-cli wiki_ingest --corpus_path ./documents --collection_name research
```

#### Using Python

```python
from libbydbot.brain.wiki import WikiManager

wiki = WikiManager(
    collection_name="research",
    wiki_base="~/.libby/wikis",
    model="llama3.2"
)

result = wiki.ingest_source(
    doc_name="research_paper.pdf",
    doc_content="The full text of the paper..."
)

print(f"Pages touched: {result['pages_touched']}")
print(f"Entities: {result['entities_created']}")
print(f"Concepts: {result['concepts_created']}")
```

### What Happens During Ingest

1. **Source Summary** — LLM reads the document and produces a structured summary
2. **Entity Extraction** — Key people, organizations, objects are identified
3. **Concept Extraction** — Topics, theories, ideas are extracted
4. **Page Creation/Update**:
   - `sources/<doc>.md` — Full summary with takeaways and questions
   - `entities/<entity>.md` — Description + links to sources mentioning it
   - `concepts/<concept>.md` — Description + links to sources discussing it
   - `synthesis/overview.md` — Running synthesis notes
5. **Index Update** — `index.md` is rebuilt with all pages
6. **Log Entry** — `log.md` gets an append-only entry

### Health-Checking the Wiki (Lint)

Run periodic lint passes to keep the wiki healthy: 

```bash
# Using TUI: Wiki Browser → Lint button
# Using CLI:
libby-cli wiki_lint --collection_name research --auto_fix
```

The lint checks for:
- **Orphan pages** — pages nobody links to
- **Broken links** — `[[wikilinks]]` pointing to missing pages
- **Contradictions** — conflicting claims between pages
- **Stale claims** — claims superseded by newer sources
- **Missing pages** — important terms mentioned but not given a page

With `--auto_fix`, stub pages are created for broken links.

### Wiki Status

```bash
libby-cli wiki_status --collection_name research
```

Shows: total pages, breakdown by category, orphans, broken links, last operation.

---

## Step 7: Querying Your Documents

### Using the TUI Chat

Navigate to **Chat** (`Ctrl+C`), select a mode, and type your question.

**Modes:**
- **RAG Answer** — Searches raw embedded documents
- **Wiki Query** — Searches the accumulated wiki (much faster for synthesis questions)

### Using the Legacy CLI

```bash
# Ask against raw documents (RAG)
libby-cli answer "What are the main findings?" --collection_name research

# Ask against the wiki
libby-cli wiki_query "What are the key concepts?" --collection_name research --file_answer
```

The `--file_answer` flag saves the response as a new page in `synthesis/`.

### Using Python

```python
from libbydbot.brain import LibbyDBot
from libbydbot.brain.wiki import WikiManager

# Initialize the bot
bot = LibbyDBot(
    name="MyBot",
    model="llama3.2",
    embed_db="duckdb:///embeddings.duckdb"
)

# RAG: Retrieve relevant documents
context = bot.DE.retrieve_docs(
    query="What are the key findings?",
    collection="research",
    num_docs=5
)

# Ask with context
bot.set_context(context)
response = bot.ask("Summarize the main findings")
print(response)

# Wiki Query
wiki = WikiManager(collection_name="research")
result = wiki.query("What are the key concepts?", file_answer=True)
print(result['answer'])
print(f"Sources: {result['sources_used']}")
print(f"Confidence: {result['confidence']}")
```

---

## Step 8: Using the REST API

### Start the Server

```bash
# Using the CLI
uv run libby-server

# With custom options
uv run libby-server --host 0.0.0.0 --port 8080 --reload
```

### API Examples

#### Embed Text

```bash
curl -X POST "http://localhost:8001/api/embed/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of AI...",
    "doc_name": "notes.txt",
    "collection_name": "research"
  }'
```

#### Upload and Embed PDF

```bash
curl -X POST "http://localhost:8001/api/embed/upload" \
  -F "file=@document.pdf" \
  -F "collection_name=research"
```

#### Retrieve Documents

```bash
curl -X POST "http://localhost:8001/api/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "collection_name": "research",
    "num_docs": 5
  }'
```

#### Wiki Ingest

```bash
curl -X POST "http://localhost:8001/api/wiki/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "paper.pdf",
    "doc_content": "The full text...",
    "collection_name": "research"
  }'
```

#### Wiki Query

```bash
curl -X POST "http://localhost:8001/api/wiki/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "collection_name": "research",
    "file_answer": false
  }'
```

#### Wiki Lint

```bash
curl -X POST "http://localhost:8001/api/wiki/lint" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "research",
    "auto_fix": true
  }'
```

#### Wiki Status

```bash
curl "http://localhost:8001/api/wiki/status/research"
```

---

## Advanced Usage

### Custom Embedding Model

```python
from libbydbot.brain.embed import DocEmbedder

# Use a specific embedding model
embedder = DocEmbedder(
    col_name="my_collection",
    dburl="duckdb:///embeddings.duckdb",
    embedding_model="mxbai-embed-large"
)
```

### Re-embedding Documents

When you change your embedding model in settings, you can re-embed existing documents:

```bash
# Using CLI - re-embed all documents with a new model
libby-cli reembed --new_model mxbai-embed-large

# Re-embed a specific collection
libby-cli reembed --collection_name research --new_model embeddinggemma

# View current embedding models in use
libby-cli model-info
```

Using Python:

```python
from libbydbot.brain.embed import DocEmbedder

embedder = DocEmbedder(col_name="my_collection", dburl="duckdb:///embeddings.duckdb")

# Run migration first (adds embedding_model column if needed)
embedder._migrate_add_embedding_model()

# Check current model info
info = embedder.get_embedding_model_info()
print(info)

# Re-embed with a new model
stats = embedder.reembed(
    collection_name="research",
    new_model="mxbai-embed-large",
    batch_size=100
)
print(f"Updated {stats['updated']}/{stats['total']} documents")
```

Using REST API:

```bash
# Re-embed via API
curl -X POST "http://localhost:8001/api/embed/reembed" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "research",
    "new_model": "mxbai-embed-large",
    "batch_size": 100
  }'

# Get model info
curl "http://localhost:8001/api/embed/model-info"
```

### Article Summarization

```python
from libbydbot.brain.analyze import ArticleSummarizer

summarizer = ArticleSummarizer(model="llama3.2")
summary = summarizer.summarize(article_text)

print(f"Title: {summary.title}")
print(f"Research Question: {summary.research_question}")
print(f"Results: {summary.results}")
```

### Memory and Chat History

```python
from libbydbot.brain.memory import History

history = History(dburl="sqlite:///memory.db")
history.memorize(
    user_id=1,
    question="What is AI?",
    response="AI is artificial intelligence...",
    context="Machine learning context"
)
```

### Direct Wiki Manager Usage

```python
from libbydbot.brain.wiki import WikiManager

wiki = WikiManager(
    collection_name="research",
    wiki_base="~/.libby/wikis",
    model="llama3.2"
)

# Ingest a source
result = wiki.ingest_source("paper.pdf", "Full text...")

# Query
result = wiki.query("What are the main concepts?", file_answer=True)

# Lint
report = wiki.lint(auto_fix=True)
print(f"Orphans: {len(report['orphan_pages'])}")
print(f"Broken links: {len(report['broken_links'])}")

# Status
status = wiki.status()
print(f"Total pages: {status['total_pages']}")
```

## Docker Deployment

```bash
# Build and run with Docker (includes Ollama server and mxbai-embed-large model)
docker build -t libby-api:latest .
docker run -d -p 8001:8000 \
  -v libby-data:/data \
  -v libby-wikis:/wikis \
  -v ollama-models:/root/.ollama \
  -e EMBED_DB=duckdb:///data/embeddings.duckdb \
  -e WIKI_BASE_PATH=/wikis \
  libby-api:latest

# Or use docker compose (recommended)
docker compose up -d
```

> **Note:**
> - Port 8001 is used to avoid conflicts. Change to `8000:8000` if port 8000 is available.
> - The Docker image includes Ollama server with the `mxbai-embed-large` embedding model pre-installed.
> - Models are persisted in the `ollama-models` volume for faster restarts.
> - Wikis are persisted in the `libby-wikis` volume mounted at `/wikis`.

## Next Steps

- **Explore the TUI** — Launch `uv run libby` and navigate through screens
- **Build a Wiki** — Ingest a collection of papers and watch the knowledge graph grow
- **Try different models** — Llama3, Gemma, GPT-4o, Qwen3, Gemini
- **Set up PostgreSQL** with pgvector for production
- **Open your wiki in Obsidian** — The markdown format is fully compatible
- **Integrate** with your existing applications via the REST API
