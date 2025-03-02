# Building a Local Knowledge Base

This tutorial will guide you through creating a local document collection and using Libby to process and index them for AI-powered retrieval and generation.

## Step 1: Setting Up Your Environment

First, ensure you have Libby installed and the required dependencies:

```bash
pip install libby
pip install pyzotero pdfminer.six
```

## Step 2: Collecting Documents

### Option 1: From Zotero Library
If you use Zotero for reference management:

```python
from pyzotero import zotero

# Initialize Zotero client
zot = zotero.Zotero(library_id, library_type, api_key)

# Get all PDF attachments
items = zot.everything(zot.top())
pdfs = [item for item in items if item['data']['contentType'] == 'application/pdf']
```

### Option 2: From Local Directory
For local PDF collections:

```bash
# Create organized directory structure
mkdir -p documents/pdfs documents/texts
```

## Step 3: Processing PDFs

Extract text from PDFs using PDFMiner:

```python
from pdfminer.high_level import extract_text

def process_pdf(pdf_path, output_dir):
    text = extract_text(pdf_path)
    output_path = os.path.join(output_dir, os.path.basename(pdf_path) + '.txt')
    with open(output_path, 'w') as f:
        f.write(text)
```

## Step 4: Ingesting Documents into Libby

Use the FileSystemIngester to process your documents:

```python
from libbydbot.brain.ingest import FileSystemIngester

ingester = FileSystemIngester(path="documents")
ingester.ingest()
```

## Step 5: Creating Embeddings

Create vector embeddings for semantic search:

```python
from libbydbot.brain.embed import DocEmbedder

embedder = DocEmbedder(col_name="my_knowledge_base")
for doc in documents:
    embedder.embed_text(
        doctext=doc['text'],
        docname=doc['name'],
        page_number=doc['page']
    )
```

## Step 6: Querying Your Knowledge Base

Now you can query your documents:

```python
results = embedder.retrieve_docs(
    query="What are the key findings?",
    num_docs=3
)

for result in results:
    print(f"Document: {result['docname']}")
    print(f"Relevance: {result['score']}")
    print(f"Content: {result['content'][:200]}...")
```

## Advanced Usage

### Using the AI Agent
You can combine document retrieval with AI generation:

```python
from libbydbot import LibbyDBot

libby = LibbyDBot()
context = "\n".join([r['content'] for r in results])
response = libby.ask(f"Based on these documents: {context}\nWhat are the main conclusions?")
print(response)
```

### Managing Memory
Libby maintains conversation history:

```python
from libbydbot.brain.memory import History

history = History()
history.add_message("user", "What is the capital of France?")
history.add_message("assistant", "The capital of France is Paris.")
```

## Next Steps
- Explore different AI models
- Set up PostgreSQL for production use
- Create custom document processing pipelines



below we use bash to move all PDF files in a directory tree to a folder called `pdfs` and then we use the `pdf2txt.py` script from the `pdfminer` library to extract the text from the PDF files. 
```bash
find . -name "*.pdf" -exec cp {} pdfs \;
cd pdfs
for f in *.pdf; do pdf2txt.py $f > $f.txt; done
```

