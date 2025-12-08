import pytest
from libbydbot.brain.embed import DocEmbedder


def test_embed_text():
    embedder = DocEmbedder("test_collection", dburl='sqlite:///embedding.db', embedding_model='mxbai-embed-large')
    embedder.embed_text('doctext1', 'docname', 1)
    edocs = embedder.get_embedded_documents()
    assert len(edocs) == 1

def test_instantiate_sqlite():
    embedder = DocEmbedder("test_collection", dburl="sqlite:///:memory:")
    assert embedder

def test_instantiate_duckdb():
    embedder = DocEmbedder("test_collection", dburl="duckdb:///embedding1.duckdb")
    assert embedder

def test_embed_duckdb_gemini():
    embedder = DocEmbedder("test_collection", dburl="duckdb:///:memory:", embedding_model="gemini-embedding-001")
    embedder.embed_text('doctext', 'docname', 1)
    assert embedder

def test_embed_duckdb_ollama():
    embedder = DocEmbedder("test_collection", dburl="duckdb:///embedding2.duckdb", embedding_model="mxbai-embed-large")
    embedder.embed_text('doctext', 'docname', 1)
    assert embedder

def test_embed_sqlite_memory():
    embedder = DocEmbedder("test_collection", dburl="sqlite:///:memory:", embedding_model='mxbai-embed-large')
    embedder.embed_text('doctext', 'docname', 1)
    edocs = embedder.get_embedded_documents()
    assert len(edocs) == 1

def test_retrieve_docs():
    embedder = DocEmbedder("test_collection", dburl='sqlite:///embedding.db', embedding_model='mxbai-embed-large')
    result = embedder.retrieve_docs('query', "test_collection" )


def test_create_embedding():
    embedder = DocEmbedder("test_collection", dburl='sqlite:///embedding.db', embedding_model='mxbai-embed-large')
    assert 'embedding_sqlite' in embedder.embeddings_list

    embedder.embed_text('Our research also sheds light on longer-term trends linking the intensity of dengue epidemics ',
                        'docname', 1)
    res = embedder.retrieve_docs('dengue', "test_collection")
    assert len(res)
    assert 'dengue' in res


def test_get_embedded_documents():
    embedder = DocEmbedder('test_collection', dburl='duckdb:///:memory:', embedding_model='mxbai-embed-large')
    
    # Embed some test documents
    embedder.embed_text("First test document", "doc1.pdf", 1)
    embedder.embed_text("Second test document", "doc2.pdf", 1)
    embedder.embed_text("Third test document", "doc1.pdf", 2)  # Same doc, different page
    
    # Get embedded documents
    embedded_docs = embedder.get_embedded_documents()
    
    # Should return list of tuples (doc_name, collection_name)
    assert isinstance(embedded_docs, list)
    assert len(embedded_docs) == 2  # Only 2 unique documents
    
    # Check that we have the expected documents
    doc_names = [doc[0] for doc in embedded_docs]
    collection_names = [doc[1] for doc in embedded_docs]
    
    assert "doc1.pdf" in doc_names
    assert "doc2.pdf" in doc_names
    assert all(col == "test_collection" for col in collection_names)

