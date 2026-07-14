import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from libbydbot.brain.embed import DocEmbedder, MODEL_MAX_CHARS, DEFAULT_MAX_EMBED_CHARS


def _postgres_available() -> bool:
    """Check if the local PostgreSQL test server is reachable."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost", database="libby", user="libby",
            password="libby123", port=5432, connect_timeout=2
        )
        conn.close()
        return True
    except Exception:
        return False


PG_AVAILABLE = _postgres_available()


@pytest.fixture(autouse=True)
def mock_embeddings():
    with patch('libbydbot.brain.embed.DocEmbedder._generate_embedding') as mocked:
        with patch('libbydbot.brain.embed.DocEmbedder._get_embedding_dimension', return_value=1024):
            mocked.return_value = np.zeros(1024).tolist()
            yield mocked

@pytest.fixture(autouse=True)
def mock_pgvector(monkeypatch):
    # Mock postgres connection if needed, but for now we focus on SQLite/DuckDB
    pass


def test_embed_text(tmp_path):
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder("test_collection", dburl=f'sqlite:///{db_path}', embedding_model='mxbai-embed-large')
    embedder.embed_text('doctext1', 'docname', 1)
    edocs = embedder.get_embedded_documents()
    assert len(edocs) == 1


def test_get_document_texts(tmp_path):
    """Reconstruct document text from the embedding table."""
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder(
        "my_coll", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
    )
    embedder.embed_text("Page one about cats.", "doc_a", 0)
    embedder.embed_text("Page two about dogs.", "doc_a", 1)
    embedder.embed_text("Solo document about fish.", "doc_b", 0)

    # All documents in the collection, ordered by page number.
    texts = embedder.get_document_texts(collection="my_coll")
    assert set(texts.keys()) == {"doc_a", "doc_b"}
    assert texts["doc_a"] == "Page one about cats.\nPage two about dogs."
    assert texts["doc_b"] == "Solo document about fish."

    # Single document filter.
    one = embedder.get_document_texts(collection="my_coll", doc_name="doc_a")
    assert list(one.keys()) == ["doc_a"]
    assert "cats" in one["doc_a"]

    # Missing document returns empty mapping.
    assert embedder.get_document_texts(collection="my_coll", doc_name="nope") == {}

@pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not available")
def test_embed_text_postgres():
    embedder = DocEmbedder("test_collection", embedding_model='mxbai-embed-large')
    embedder.embed_text('doctext1', 'docname', 1)
    edocs = embedder.get_embedded_documents()
    assert len(edocs) >= 1

def test_instantiate_sqlite():
    embedder = DocEmbedder("test_collection", dburl="sqlite:///:memory:")
    assert embedder

def test_instantiate_duckdb(tmp_path):
    db_path = tmp_path / "embeddings.duckdb"
    embedder = DocEmbedder("test_collection", dburl=f"duckdb:///{db_path}")
    assert embedder

def test_embed_duckdb_gemini():
    embedder = DocEmbedder("test_collection", dburl="duckdb:///:memory:", embedding_model="gemini-embedding-001")
    embedder.embed_text('doctext', 'docname', 1)
    assert embedder

@pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not available")
def test_embed_postgres_gemini():
    embedder = DocEmbedder("test_collection", embedding_model="gemini-embedding-001")
    embedder.embed_text('doctext', 'docname', 1)
    assert embedder

def test_embed_duckdb_ollama(tmp_path):
    db_path = tmp_path / "embeddings.duckdb"
    embedder = DocEmbedder("test_collection", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large")
    embedder.embed_text('doctext', 'docname', 1)
    assert embedder

def test_embed_sqlite_memory():
    embedder = DocEmbedder("test_collection", dburl="sqlite:///:memory:", embedding_model='mxbai-embed-large')
    embedder.embed_text('doctext', 'docname', 1)
    edocs = embedder.get_embedded_documents()
    assert len(edocs) == 1

def test_retrieve_docs(tmp_path):
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder("test_collection", dburl=f'sqlite:///{db_path}', embedding_model='mxbai-embed-large')
    result = embedder.retrieve_docs('query', "test_collection" )


def test_create_embedding(tmp_path):
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder("test_collection", dburl=f'sqlite:///{db_path}', embedding_model='mxbai-embed-large')
    assert 'embedding_sqlite' in embedder.embeddings_list

    embedder.embed_text('Our research also sheds light on longer-term trends linking the intensity of dengue epidemics ',
                        'docname', 1)
    res = embedder.retrieve_docs('dengue', "test_collection")
    assert len(res)
    assert 'dengue' in res

@pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not available")
def test_create_embedding_postgres():
    embedder = DocEmbedder("test_collection", embedding_model='mxbai-embed-large')
    assert 'embedding' in embedder.embeddings_list

    embedder.embed_text('Our research also sheds light on longer-term trends linking the intensity of dengue epidemics ',
                        'docname', 1)
    res = embedder.retrieve_docs('dengue', "test_collection")
    assert len(res)
    assert 'dengue' in res


def test_get_embedded_documents(tmp_path):
    db_path = tmp_path / "embeddings.duckdb"
    embedder = DocEmbedder('test_collection', dburl=f'duckdb:///{db_path}', embedding_model='mxbai-embed-large')
    
    # Embed some test documents
    embedder.embed_text("First test document", "doc1.pdf", 1)
    embedder.embed_text("Second test document", "doc2.pdf", 1)
    embedder.embed_text("Third test document", "doc1.pdf", 2)  # Same doc, different page
    
    # Get embedded documents
    embedded_docs = embedder.get_embedded_documents()
    
    # Should return list of tuples (doc_name, collection_name)
    assert isinstance(embedded_docs, list)
    assert len(embedded_docs) >= 2  # At least 2 unique documents found in this test
    
    # Check that we have the expected documents
    doc_names = [doc[0] for doc in embedded_docs]
    collection_names = [doc[1] for doc in embedded_docs]
    assert "doc1.pdf" in doc_names
    assert "doc2.pdf" in doc_names
    assert all(col == "test_collection" for col in collection_names)


def test_auto_split_oversized_text(tmp_path):
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder("test_collection", dburl=f'sqlite:///{db_path}', embedding_model='mxbai-embed-large')
    limit = embedder._max_embed_chars()
    paragraphs = [f"Unique paragraph number {i}. " + "x" * 200 for i in range(80)]
    oversized_text = "\n\n".join(paragraphs)
    assert len(oversized_text) > limit
    embedder.embed_text(oversized_text, "big_doc", 1)
    docs = embedder.get_embedded_documents()
    assert len(docs) >= 1


def test_normal_text_not_auto_split(tmp_path):
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder("test_collection", dburl=f'sqlite:///{db_path}', embedding_model='mxbai-embed-large')
    normal_text = "This is a normal sized document."
    embedder.embed_text(normal_text, "small_doc", 1)
    docs = embedder.get_embedded_documents()
    assert len(docs) == 1


def test_max_embed_chars_model_specific():
    assert MODEL_MAX_CHARS["mxbai-embed-large"] == 800
    assert MODEL_MAX_CHARS["embeddinggemma"] == 8000


def test_max_embed_chars_default():
    assert DEFAULT_MAX_EMBED_CHARS == 800


@pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not available")
def test_get_embedded_documents_postgres():
    embedder = DocEmbedder('test_collection', embedding_model='mxbai-embed-large')

    # Embed some test documents
    embedder.embed_text("First test document", "doc1.pdf", 1)
    embedder.embed_text("Second test document", "doc2.pdf", 1)
    embedder.embed_text("Third test document", "doc1.pdf", 2)  # Same doc, different page

    # Get embedded documents
    embedded_docs = embedder.get_embedded_documents()

    # Should return list of tuples (doc_name, collection_name)
    assert isinstance(embedded_docs, list)
    assert len(embedded_docs) >= 2  # At least 2 unique documents

    # Check that we have the expected documents
    doc_names = [doc[0] for doc in embedded_docs]
    collection_names = [doc[1] for doc in embedded_docs]

    assert "doc1.pdf" in doc_names
    assert "doc2.pdf" in doc_names
    assert all(col == "test_collection" for col in collection_names)


class TestReconstructDocuments:
    def test_single_doc_single_chunk(self):
        rows = [(1, "col_a", "doc1", 0, "hash1", "Hello world")]
        result = DocEmbedder._reconstruct_documents(rows)
        assert ("col_a", "doc1") in result
        assert result[("col_a", "doc1")] == "Hello world"

    def test_single_doc_multiple_pages(self):
        rows = [
            (1, "col_a", "doc1", 0, "h1", "Page 0"),
            (2, "col_a", "doc1", 1, "h2", "Page 1"),
            (3, "col_a", "doc1", 2, "h3", "Page 2"),
        ]
        result = DocEmbedder._reconstruct_documents(rows)
        assert result[("col_a", "doc1")] == "Page 0\nPage 1\nPage 2"

    def test_multiple_docs_kept_separate(self):
        rows = [
            (1, "col_a", "doc1", 0, "h1", "A"),
            (2, "col_a", "doc2", 0, "h2", "B"),
            (3, "col_a", "doc1", 1, "h3", "C"),
        ]
        result = DocEmbedder._reconstruct_documents(rows)
        assert result[("col_a", "doc1")] == "A\nC"
        assert result[("col_a", "doc2")] == "B"

    def test_multiple_collections(self):
        rows = [
            (1, "col_a", "doc1", 0, "h1", "AAA"),
            (2, "col_b", "doc1", 0, "h2", "BBB"),
        ]
        result = DocEmbedder._reconstruct_documents(rows)
        assert result[("col_a", "doc1")] == "AAA"
        assert result[("col_b", "doc1")] == "BBB"

    def test_empty_rows(self):
        result = DocEmbedder._reconstruct_documents([])
        assert result == {}

    def test_sorted_by_page_then_id(self):
        rows = [
            (5, "col", "doc", 1, "h5", "page1_id5"),
            (2, "col", "doc", 0, "h2", "page0_id2"),
            (3, "col", "doc", 1, "h3", "page1_id3"),
            (1, "col", "doc", 0, "h1", "page0_id1"),
        ]
        result = DocEmbedder._reconstruct_documents(rows)
        assert result[("col", "doc")] == "page0_id1\npage0_id2\npage1_id3\npage1_id5"


def test_rechunk_sqlite_writes_shadow_collection(tmp_path):
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder("my_collection", dburl=f'sqlite:///{db_path}', embedding_model='mxbai-embed-large')

    paragraphs = [f"Paragraph {i}. " + "x" * 80 for i in range(20)]
    for i, p in enumerate(paragraphs):
        embedder.embed_text(p, "doc1.pdf", i)

    old_docs = embedder.get_embedded_documents()
    assert any(col == "my_collection" for _, col in old_docs)

    stats = embedder.reembed(
        collection_name="my_collection",
        new_model="mxbai-embed-large",
        rechunk=True,
        new_chunk_size=800,
        new_chunk_overlap=100,
    )

    assert stats["total_old_chunks"] == 20
    assert stats["total_new_chunks"] > 0
    assert stats["total_new_chunks"] < stats["total_old_chunks"]
    assert stats["shadow_collection"] == "my_collection_v2"

    # Verify shadow exists via direct query (get_embedded_documents filters _v2 out)
    shadow_count = embedder._verify_query(
        "embedding_sqlite",
        "SELECT COUNT(*) FROM embedding_sqlite WHERE collection_name='my_collection_v2'",
    )[0][0]
    assert shadow_count > 0

    # Verify get_embedded_documents filters shadow collections
    all_docs = embedder.get_embedded_documents()
    assert all(c != "my_collection_v2" for _, c in all_docs)


def test_rechunk_duckdb_writes_shadow_collection(tmp_path):
    db_path = tmp_path / "embeddings.duckdb"
    embedder = DocEmbedder("my_collection", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large")

    paragraphs = [f"Paragraph {i}. " + "x" * 80 for i in range(20)]
    for i, p in enumerate(paragraphs):
        embedder.embed_text(p, "doc1.pdf", i)

    stats = embedder.reembed(
        collection_name="my_collection",
        new_model="mxbai-embed-large",
        rechunk=True,
        new_chunk_size=800,
        new_chunk_overlap=100,
    )

    assert stats["total_old_chunks"] == 20
    assert stats["total_new_chunks"] > 0
    assert stats["total_new_chunks"] < stats["total_old_chunks"]
    assert stats["shadow_collection"] == "my_collection_v2"


def test_rechunk_preserves_text(tmp_path):
    db_path = tmp_path / "embedding.db"
    embedder = DocEmbedder("my_collection", dburl=f'sqlite:///{db_path}', embedding_model='mxbai-embed-large')

    chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Python is a versatile programming language.",
    ]
    for i, chunk in enumerate(chunks):
        embedder.embed_text(chunk, "doc1.pdf", i)

    all_original_text = "\n".join(chunks)

    stats = embedder.reembed(
        collection_name="my_collection",
        rechunk=True,
        new_chunk_size=500,
        new_chunk_overlap=50,
    )

    assert stats["total_new_chunks"] >= 1

    with embedder.connection as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT document FROM {embedder.table_name} WHERE collection_name = ?",
            ("my_collection_v2",),
        )
        new_chunks = [row[0] for row in cursor.fetchall()]

    reconstructed = "\n".join(new_chunks)
    for original_word in all_original_text.split():
        assert original_word in reconstructed


def test_reembed_without_rechunk_no_rechunk_stats(tmp_path):
    db_path = tmp_path / "embeddings.duckdb"
    embedder = DocEmbedder("my_collection", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large")

    embedder.embed_text("Hello world", "doc1.pdf", 0)
    embedder.embed_text("Second chunk", "doc1.pdf", 1)

    stats = embedder.reembed(
        collection_name="my_collection",
        rechunk=False,
    )

    assert stats["total"] == 2
    assert stats["updated"] == 2
    assert "total_old_chunks" not in stats
    assert "shadow_collection" not in stats

