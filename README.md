# Libby D. Bot

[![DOI](https://zenodo.org/badge/784398327.svg)](https://zenodo.org/doi/10.5281/zenodo.12744747)

Libby the librarian. AI agent specialized in creating and querying embeddings for RAG (Retrieval Augmented Generation).
![Libby D. Bot](/libby.jpeg)

## Installation

You can install Libby D. Bot using pip:

```bash
pip install -U libby
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
libby generate "Write a summary of..." --output_file output.txt
```

## Features

- Multiple language support (English and Portuguese)
- Various AI models available (Llama3, Gemma, ChatGPT)
- PDF document processing and embedding
- Question answering with context from your documents
- Content generation capabilities

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
