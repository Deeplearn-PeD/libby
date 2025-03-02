# API Reference

## LibbyDBot Class

The main AI agent class that provides document processing and question answering capabilities.

### Initialization
```python
from libbydbot.brain import LibbyDBot

# Create instance with default settings
bot = LibbyDBot()

# Custom initialization
bot = LibbyDBot(
    name="Custom Name",
    languages=['pt_BR', 'en'],
    model='gpt-4o'
)
```

### Core Methods

#### set_context(context: str)
Sets the context for the next question/answer interaction.

#### set_prompt(prompt_template: str)
Sets a custom prompt template for the AI model.

#### ask(question: str, user_id: int = 1) -> str
Asks a question and returns the AI's response.

Example:
```python
response = bot.ask("What is the capital of France?")
```

### Configuration

#### Supported Models
Any models that you have API keys for, plus all local models accessible via Ollama.
Try:
```python
bot.llm.available_models()
```

#### Supported Languages
- Portuguese (pt_BR)
- English (en)

### Memory Management
LibbyDBot automatically maintains conversation history through the Memory class.

### Integration with Other Components

#### Document Processing
```python
from libbydbot.brain.ingest import FileSystemIngester
ingester = FileSystemIngester(path="documents")
ingester.ingest()
```

#### Document Embedding
```python
from libbydbot.brain.embed import DocEmbedder
embedder = DocEmbedder(col_name="my_collection")
```

#### Article Summarization
```python
from libbydbot.brain.analyze import ArticleSummarizer
summarizer = ArticleSummarizer()
summary = summarizer.summarize(article_text)
```
