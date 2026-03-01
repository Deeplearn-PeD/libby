# Tutorial

This tutorial will guide you through using Libby D. Bot to create embeddings from your documents and query them.

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
```

## Step 3: Preparing Your Documents

Place your PDF documents in a directory:

```bash
mkdir -p documents
# Copy your PDFs to the documents folder
```

## Step 4: Creating Embeddings

Use the CLI to embed your documents:

```bash
# Basic usage
libby embed --corpus_path ./documents

# With custom collection name
libby embed --corpus_path ./documents --collection_name research
```

The embedding process:
1. Extracts text from PDFs using PyMuPDF
2. Splits text into chunks (default: 800 chars with 100 char overlap)
3. Generates vector embeddings using the default model (embeddinggemma)
4. Stores embeddings in the database

## Step 5: Querying Your Documents

### Using the CLI

```bash
# Ask a question
libby answer "What are the main findings?"

# Specify a collection
libby answer "What are the main findings?" --collection_name research
```

### Using Python API

```python
from libbydbot.brain import LibbyDBot
from libbydbot.brain.embed import DocEmbedder

# Initialize the bot
bot = LibbyDBot(
    name="MyBot",
    model="llama3.2",
    embed_db="duckdb:///embeddings.duckdb"
)

# Retrieve relevant documents
context = bot.DE.retrieve_docs(
    query="What are the key findings?",
    collection="research",
    num_docs=5
)

# Ask a question with context
bot.set_context(context)
response = bot.ask("Summarize the main findings")
print(response)
```

## Step 6: Using the REST API

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
curl -X POST "http://localhost:8000/api/embed/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of AI...",
    "doc_name": "notes.txt",
    "collection_name": "research"
  }'
```

#### Upload and Embed PDF

```bash
curl -X POST "http://localhost:8000/api/embed/upload" \
  -F "file=@document.pdf" \
  -F "collection_name=research"
```

#### Retrieve Documents

```bash
curl -X POST "http://localhost:8000/api/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "collection_name": "research",
    "num_docs": 5
  }'
```

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

## Docker Deployment

```bash
# Build and run
docker build -t libby-api:latest .
docker run -d -p 8000:8000 \
  -v libby-data:/data \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  libby-api:latest

# Or use docker-compose
docker-compose up -d
```

## Next Steps

- Explore different AI models (Llama3, Gemma, GPT-4o, Qwen3)
- Set up PostgreSQL with pgvector for production
- Create custom document processing pipelines
- Integrate with your existing applications via the REST API
