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

| Name | Code | Default |
|------|------|---------|
| GemmaEmbedding | `embeddinggemma` | Yes |
| Mxbai | `mxbai-embed-large` | No |
| Gemini | `gemini-embedding-001` | No |

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/embed/text` | Embed raw text |
| `POST` | `/api/embed/upload` | Upload and embed PDF |
| `POST` | `/api/retrieve` | Hybrid search |
| `GET` | `/api/documents` | List embedded documents |
| `GET` | `/api/collections` | List collections |
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

# Start API server
libby-server --host 0.0.0.0 --port 8000
```

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
