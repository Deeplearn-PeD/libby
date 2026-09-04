# API Reference

## Core Classes

### LibbyDBot

The main AI agent class providing document processing and question answering.

```python
from libbydbot.brain import LibbyDBot

bot = LibbyDBot(
    name="Libby D. Bot",
    languages=['pt_BR', 'en'],
    model='llama3.2',
    dburl='sqlite:///memory.db',
    embed_db='duckdb:///embeddings.duckdb'
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"Libby D. Bot"` | Bot name |
| `languages` | `list[str]` | `['pt_BR', 'en']` | Supported languages |
| `model` | `str` | `"llama3.2"` | LLM model to use |
| `dburl` | `str` | `"sqlite:///memory.db"` | Memory database URL |
| `embed_db` | `str` | `"duckdb:///embeddings.duckdb"` | Embeddings database URL |

#### Methods

##### `ask(question: str, user_id: int = 1) -> str`

Ask a question and get a response.

```python
response = bot.ask("What is the main topic?")
```

##### `set_context(context: str)`

Set the context for the next question.

```python
bot.set_context("This is background information...")
```

##### `set_prompt(prompt_template: str)`

Set a custom prompt template.

```python
bot.set_prompt("You are a helpful research assistant.")
```

---

### DocEmbedder

Handles document embedding and retrieval.

```python
from libbydbot.brain.embed import DocEmbedder

embedder = DocEmbedder(
    col_name="my_collection",
    dburl="duckdb:///embeddings.duckdb",
    embedding_model="embeddinggemma",
    chunk_size=800,
    chunk_overlap=100
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `col_name` | `str` | Required | Collection name |
| `dburl` | `str` | `os.getenv("PGURL")` | Database URL |
| `embedding_model` | `str \| None` | From settings | Embedding model |
| `chunk_size` | `int` | `800` | Text chunk size |
| `chunk_overlap` | `int` | `100` | Chunk overlap |

#### Methods

##### `embed_text(doctext: str, docname: str, page_number: int)`

Embed a single text document.

```python
embedder.embed_text(
    doctext="Document content...",
    docname="document.pdf",
    page_number=1
)
```

##### `embed_path(corpus_path: str)`

Embed all PDFs in a directory.

```python
embedder.embed_path("/path/to/documents")
```

##### `retrieve_docs(query: str, collection: str = "", num_docs: int = 5) -> str`

Retrieve documents using hybrid search.

```python
results = embedder.retrieve_docs(
    query="What is machine learning?",
    collection="research",
    num_docs=5
)
```

##### `retrieve_docs_with_metadata(query: str, collection: str = "", num_docs: int = 5) -> list[dict]`

Retrieve documents with metadata.

```python
results = embedder.retrieve_docs_with_metadata(
    query="What is machine learning?",
    collection="research",
    num_docs=5
)
for r in results:
    print(f"{r['doc_name']} (p.{r['page_number']}): {r['score']}")
```

##### `get_embedded_documents() -> list[tuple]`

List all embedded documents.

```python
docs = embedder.get_embedded_documents()
for doc_name, collection in docs:
    print(f"{doc_name} in {collection}")
```

##### `get_document_chunks(doc_name: str, collection: str = "") -> list[dict]`

Get all embedded chunks of a document, ordered by page/chunk number. Used by the
knowledge graph to link chunks to the entities and concepts they mention.

```python
chunks = embedder.get_document_chunks("paper.pdf", collection="research")
for chunk in chunks:
    print(chunk["page_number"], chunk["doc_hash"], chunk["content"][:60])
```

##### `reembed(collection_name: str = "", new_model: str | None = None, batch_size: int = 100) -> dict`

Re-embed documents with a new embedding model.

```python
# Re-embed all documents with a new model
stats = embedder.reembed(new_model="mxbai-embed-large")

# Re-embed specific collection
stats = embedder.reembed(
    collection_name="research",
    new_model="embeddinggemma",
    batch_size=100
)
print(f"Updated {stats['updated']}/{stats['total']} documents")
print(f"Old model: {stats['old_model']}, New model: {stats['new_model']}")
```

##### `get_embedding_model_info() -> dict`

Get information about embedding models used in the database.

```python
info = embedder.get_embedding_model_info()
print(f"Total documents: {info['total_documents']}")
for model, collections in info['models'].items():
    for collection, count in collections.items():
        print(f"  {model}: {count} docs in '{collection}'")
```

##### `_migrate_add_embedding_model()`

Migrate existing database to add embedding_model column.

```python
# Run this first if you have an existing database
embedder._migrate_add_embedding_model()
```

---

### WikiManager

Manages the LLM Wiki and its knowledge graph for a single collection. The wiki is a
directory of Obsidian-compatible markdown pages; the knowledge graph is a NetworkX
graph persisted as `graph.json` inside that directory.

```python
from libbydbot.brain.wiki import WikiManager

wiki = WikiManager(
    collection_name="research",
    wiki_base="~/.libby/wikis",
    model="llama3.2",
    graph_enabled=True  # None reads the WIKI_GRAPH_ENABLED setting
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | Required | Collection whose wiki to manage |
| `wiki_base` | `str \| Path` | `~/.libby/wikis` | Base directory for wikis |
| `model` | `str` | `"kimi-k2.5"` | LLM model for summaries/synthesis |
| `graph_enabled` | `bool \| None` | From settings | Maintain the knowledge graph |

#### Wiki Methods

##### `ingest_source(doc_name: str, doc_content: str, source_type: str = "document") -> dict`

Summarize a source, write/update wiki pages, and update the knowledge graph
(including links from the document's embedded chunks to extracted entities/concepts).

##### `query(question: str, file_answer: bool = False) -> dict`

Answer a question from the wiki. Pages are ranked with the knowledge graph when one
exists (seed matching + neighborhood expansion + degree); otherwise a keyword
heuristic is used.

##### `lint(auto_fix: bool = False) -> dict`

Health-check: orphans, broken links, contradictions, stale claims, missing pages,
and graph hubs.

##### `status() -> dict`

Wiki statistics, including a `graph` key with knowledge graph statistics when a
graph exists.

#### Knowledge Graph Methods

##### `graph_rebuild() -> dict`

Rebuild the graph from the wiki pages on disk. Chunk nodes and their reference
edges are preserved. Returns graph statistics.

##### `graph_status() -> dict`

Graph statistics: node/edge counts by type and the most-connected hub nodes.

##### `graph_path(a: str, b: str) -> dict`

Shortest undirected path between two concepts (fuzzy name resolution).

```python
result = wiki.graph_path("Alice", "Gravity")
if result["found"]:
    print(" -> ".join(n["title"] for n in result["nodes"]))
```

##### `graph_explain(name: str) -> dict`

A node's attributes plus its inbound/outbound connections with edge types.

##### `graph_query(question: str, max_nodes: int = 15) -> dict`

Ranked subgraph (nodes + edges) relevant to a question.

##### `graph_export_html(path: str | Path | None = None) -> Path`

Export an interactive pyvis visualization. Defaults to `graph.html` in the wiki
directory.

---

### WikiKnowledgeGraph

The graph engine used by `WikiManager` (`libbydbot.brain.graph`). Use it directly
for low-level control.

```python
from libbydbot.brain.graph import WikiKnowledgeGraph

kg = WikiKnowledgeGraph(wiki_dir="~/.libby/wikis/research",
                        collection_name="research")
kg.rebuild()
```

- **Nodes** — page nodes identified by relative path (`entities/alice.md`) with
  `node_type` of `source`, `entity`, `concept`, `synthesis`, or `stub`; chunk
  nodes identified by `chunk:<doc_hash>`.
- **Edge types** — `mentions`, `mentioned_in`, `related_to`, `links_to`,
  `chunk_of` (chunk → source), `references` (chunk → entity/concept matched by name).

#### Key Methods

| Method | Description |
|--------|-------------|
| `rebuild()` | Rebuild page nodes/edges from disk; chunks survive |
| `update_from_ingest(doc_name, summary, chunks)` | Incremental update after an ingest |
| `add_chunks(chunks)` / `link_chunks_to_names(chunks, names)` | Chunk node management |
| `save()` / `load()` | Persist to / load from `graph.json` |
| `shortest_path(a, b)` / `explain(name)` | Navigation queries |
| `score_pages(question)` / `subgraph_for_query(question)` | Query ranking |
| `hubs(top_n)` / `status()` | Centrality and statistics |
| `export_html(path)` | Interactive pyvis visualization |

---

### PDFPipeline

Iterator for processing PDF files.

```python
from libbydbot.brain.ingest import PDFPipeline

pipeline = PDFPipeline(
    path="/path/to/pdfs",
    chunk_size=800,
    chunk_overlap=100
)

for text, metadata in pipeline:
    print(f"Processing: {metadata.get('title')}")
```

---

### ArticleSummarizer

Summarize scientific articles using structured output.

```python
from libbydbot.brain.analyze import ArticleSummarizer

summarizer = ArticleSummarizer(model="llama3.2")
summary = summarizer.summarize(article_text)

print(summary.title)
print(summary.research_question)
print(summary.keywords)
print(summary.results)
print(summary.conclusions)
```

#### ArticleSummary Fields

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Article title |
| `summary` | `str` | Article summary |
| `research_question` | `str` | Main research question |
| `keywords` | `list[str]` | List of keywords |
| `results` | `list[str]` | List of results |
| `conclusions` | `list[str]` | List of conclusions |

---

### History

Manages conversation history persistence.

```python
from libbydbot.brain.memory import History

history = History(dburl="sqlite:///memory.db")

# Store a conversation
history.memorize(
    user_id=1,
    question="What is AI?",
    response="AI is artificial intelligence...",
    context="Background context"
)

# Recall conversations
conversations = history.recall(user_id=1)
```

---

### Settings

Configuration management using pydantic-settings.

```python
from libbydbot.settings import Settings

settings = Settings()

# Get default LLM model
print(settings.default_model)  # "llama3.2"

# Get default embedding model
print(settings.default_embedding_model)  # "embeddinggemma"
```

#### Available LLM Models

| Name | Code | Default |
|------|------|---------|
| Llama3 | `llama3.2` | Yes |
| Gemma | `gemma3` | No |
| ChatGPT | `gpt-4o` | No |
| Qwen | `qwen3` | No |

#### Available Embedding Models

| Name | Code | Dimension | Default |
|------|------|-----------|---------|
| GemmaEmbedding | `embeddinggemma` | 768 | Yes |
| Mxbai | `mxbai-embed-large` | 1024 | No |
| Gemini | `gemini-embedding-001` | 1024 | No |

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/embed/text` | Embed raw text |
| `POST` | `/api/embed/upload` | Upload and embed PDF |
| `POST` | `/api/embed/reembed` | Re-embed documents with new model |
| `GET` | `/api/embed/model-info` | Get embedding model info |
| `POST` | `/api/retrieve` | Hybrid search |
| `GET` | `/api/documents` | List embedded documents |
| `GET` | `/api/collections` | List collections |
| `POST` | `/api/wiki/ingest` | Ingest a source into the wiki |
| `POST` | `/api/wiki/query` | Query the wiki |
| `POST` | `/api/wiki/lint` | Lint the wiki |
| `GET` | `/api/wiki/status/{collection_name}` | Wiki statistics |
| `POST` | `/api/wiki/graph/rebuild` | Rebuild the knowledge graph |
| `GET` | `/api/wiki/graph/{collection_name}` | Knowledge graph statistics |
| `GET` | `/api/wiki/graph/{collection_name}/viz` | Serve the interactive `graph.html` visualization |
| `POST` | `/api/wiki/graph/path` | Shortest path between two nodes |
| `POST` | `/api/wiki/graph/explain` | Explain a node's connections |
| `POST` | `/api/wiki/graph/query` | Ranked subgraph for a question |
| `GET` | `/api/health` | Health check |

---

## CLI Commands

```bash
# Embed documents
libby embed --corpus_path ./docs --collection_name my_collection

# Answer questions
libby answer "Your question?" --collection_name my_collection

# Generate content
libby generate "Your prompt" --output_file output.txt

# Re-embed documents with a new model
libby reembed --new_model mxbai-embed-large --collection_name research

# View embedding model info
libby model-info

# Wiki: ingest, query, lint, status
libby-cli wiki_ingest --corpus_path ./docs --collection_name research
libby-cli wiki_query "What is the main topic?" --collection_name research
libby-cli wiki_lint --collection_name research --auto_fix
libby-cli wiki_status --collection_name research

# Wiki knowledge graph: rebuild/stats, path, explain, visualization export
libby-cli wiki_graph research
libby-cli wiki_path "Alice" "Gravity" research
libby-cli wiki_explain "Alice" research
libby-cli wiki_graph_export research --output graph.html

# Start API server
libby-server --host 0.0.0.0 --port 8000
```

!!! note
    For the wiki commands above, pass the collection as a positional argument:
    `--collection_name` is also a constructor parameter of the CLI class, so a flag
    is consumed by the constructor rather than the command.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `PGURL` | PostgreSQL connection URL | - |
| `EMBED_DB` | Embedding database URL | `duckdb:///data/embeddings.duckdb` |
| `EMBEDDING_MODEL` | Embedding model | `embeddinggemma` |
| `COLLECTION_NAME` | Default collection | `main` |
| `WIKI_BASE_PATH` | Base directory for LLM wikis | `~/.libby/wikis` |
| `WIKI_AUTO_INGEST` | Auto-ingest into wiki after embedding | `False` |
| `WIKI_GRAPH_ENABLED` | Maintain the knowledge graph over wiki + chunks | `True` |
