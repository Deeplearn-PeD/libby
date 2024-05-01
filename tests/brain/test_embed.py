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


@patch('libbydbot.brain.embed.ollama')
def test_retrieve_docs(mock_ollama):
    mock_ollama.embeddings.return_value = {"embedding": [1, 2, 3]}
    embedder = DocEmbedder(name='embeddings2')
    embedder.session = MagicMock()
    embedder.session.scalars.return_value = ['doc1', 'doc2', 'doc3']
    result = embedder.retrieve_docs('query')
    assert result == 'doc1\ndoc2\ndoc3'

def test_create_embedding():
    embedder = DocEmbedder(name='test_embeddings', create=True)
    assert embedder.embeddings_list == ['test_embeddings']
    embedder.set_schema('test_embeddings')
    assert embedder.schema is not None
    embedder.session = MagicMock()
    embedder.embed_text('Our research also sheds light on longer-term trends linking the intensity of dengue epidemics ',
                        'docname', 1)
    embedder.retrieve_docs('query')

