import pytest
from libbydbot.brain.embed import DocEmbedder


def test_embed_text():
    embedder = DocEmbedder("test_collection")
    embedder.embed_text('doctext', 'docname', 1)

def test_instantiate():
    embedder = DocEmbedder("test_collection", dburl="sqlite:///memory.db")
    assert embedder

def test_retrieve_docs():
    embedder = DocEmbedder("test_collection")
    result = embedder.retrieve_docs('query', "test_collection" )
    assert result

def test_create_embedding():
    embedder = DocEmbedder("test_collection", create=True)
    assert 'embedding' in embedder.embeddings_list
    assert embedder.embedding is not None
    embedder.embed_text('Our research also sheds light on longer-term trends linking the intensity of dengue epidemics ',
                        'docname', 1)
    res = embedder.retrieve_docs('dengue', "test_collection")
    assert len(res)
    assert 'dengue' in res

