import pytest
from unittest.mock import patch, MagicMock
from libbydbot.brain.embed import DocEmbedder

@patch('libbydbot.brain.embed.ollama')
@patch('libbydbot.brain.embed.sha256')
def test_embed_text(mock_sha256, mock_ollama):
    mock_ollama.embeddings.return_value = {"embedding": [1, 2, 3]}
    mock_sha256.return_value.hexdigest.return_value = 'hash'
    embedder = DocEmbedder()
    embedder.session = MagicMock()
    embedder.embed_text('doctext', 'docname', 1)
    embedder.session.add.assert_called_once()
    embedder.session.commit.assert_called_once()

@patch('libbydbot.brain.embed.ollama')
def test_retrieve_docs(mock_ollama):
    mock_ollama.embeddings.return_value = {"embedding": [1, 2, 3]}
    embedder = DocEmbedder()
    embedder.session = MagicMock()
    embedder.session.scalars.return_value = ['doc1', 'doc2', 'doc3']
    result = embedder.retrieve_docs('query')
    assert result == 'doc1\ndoc2\ndoc3'

@patch('libbydbot.brain.embed.ollama')
def test_generate_response(mock_ollama):
    mock_ollama.generate.return_value = {"response": "response"}
    embedder = DocEmbedder()
    embedder.retrieve_docs = MagicMock()
    embedder.retrieve_docs.return_value = 'context'
    result = embedder.generate_response('question')
    assert result == 'response'