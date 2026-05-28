import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from libbydbot.brain.embed import DocEmbedder


@pytest.fixture(autouse=True)
def mock_embeddings():
    with patch("libbydbot.brain.embed.DocEmbedder._generate_embedding") as mocked:
        mocked.return_value = np.zeros(1024).tolist()
        yield mocked


class TestListBackends:
    def test_sqlite_source_lists_all_backends(self, tmp_path):
        db_path = tmp_path / "embeddings.db"
        embedder = DocEmbedder(
            "test_col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        backends = embedder.list_backends()

        assert len(backends) == 3
        names = [b["name"] for b in backends]
        assert "postgresql" in names
        assert "duckdb" in names
        assert "sqlite" in names

        sqlite_be = next(b for b in backends if b["name"] == "sqlite")
        assert sqlite_be["is_current"] is True
        assert sqlite_be["is_configured"] is True

        pg_be = next(b for b in backends if b["name"] == "postgresql")
        assert pg_be["is_current"] is False

    def test_duckdb_source_lists_duckdb_as_current(self, tmp_path):
        db_path = tmp_path / "embeddings.duckdb"
        embedder = DocEmbedder(
            "test_col", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large"
        )
        backends = embedder.list_backends()

        duckdb_be = next(b for b in backends if b["name"] == "duckdb")
        assert duckdb_be["is_current"] is True

        sqlite_be = next(b for b in backends if b["name"] == "sqlite")
        assert sqlite_be["is_current"] is False

    def test_safe_location_strips_credentials(self):
        result = DocEmbedder._safe_location(
            "postgresql://user:secret@db.example.com:5432/mydb"
        )
        assert "secret" not in result
        assert "db.example.com" in result
        assert "5432" in result

    def test_safe_location_empty_string(self):
        result = DocEmbedder._safe_location("")
        assert result == ""


class TestMigrateBackendSqliteToDuckdb:
    def test_basic_migration(self, tmp_path):
        src_db = tmp_path / "source.db"
        tgt_db = tmp_path / "target.duckdb"

        src = DocEmbedder(
            "col_a", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )
        src.embed_text("Text one about cats", "doc1.pdf", 0)
        src.embed_text("Text two about dogs", "doc1.pdf", 1)
        src.embed_text("Text three about birds", "doc2.pdf", 0)

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            stats = src.migrate_backend(
                target_backend="duckdb",
                collection_name="col_a",
                batch_size=10,
            )

        assert stats["success"] is True
        assert stats["migrated"] == 3
        assert stats["skipped"] == 0
        assert stats["source_backend"] == "sqlite"
        assert stats["target_backend"] == "duckdb"
        assert len(stats["errors"]) == 0

        tgt = DocEmbedder(
            "col_a", dburl=f"duckdb:///{tgt_db}", embedding_model="mxbai-embed-large"
        )
        count = tgt._count_source_records("col_a")
        assert count == 3

    def test_migration_all_collections(self, tmp_path):
        src_db = tmp_path / "source.db"
        tgt_db = tmp_path / "target.duckdb"

        src = DocEmbedder(
            "col_a", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )
        src.embed_text("Text A1", "doc1.pdf", 0)
        src.embed_text("Text A2", "doc2.pdf", 0)

        src2 = DocEmbedder(
            "col_b", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )
        src2.embed_text("Text B1", "doc3.pdf", 0)

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            stats = src.migrate_backend(
                target_backend="duckdb",
                collection_name="",
                batch_size=10,
            )

        assert stats["success"] is True
        assert stats["migrated"] == 3

    def test_dry_run_does_not_write(self, tmp_path):
        src_db = tmp_path / "source.db"
        tgt_db = tmp_path / "target.duckdb"

        src = DocEmbedder(
            "col_a", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )
        src.embed_text("Some text", "doc1.pdf", 0)

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            stats = src.migrate_backend(
                target_backend="duckdb",
                dry_run=True,
            )

        assert stats["success"] is True
        assert stats["migrated"] == 1

        tgt = DocEmbedder(
            "col_a", dburl=f"duckdb:///{tgt_db}", embedding_model="mxbai-embed-large"
        )
        count = tgt._count_source_records("col_a")
        assert count == 0


class TestMigrateBackendDuckdbToSqlite:
    def test_basic_migration(self, tmp_path):
        src_db = tmp_path / "source.duckdb"
        tgt_db = tmp_path / "target.db"

        src = DocEmbedder(
            "col_a", dburl=f"duckdb:///{src_db}", embedding_model="mxbai-embed-large"
        )
        src.embed_text("DuckDB text one", "doc1.pdf", 0)
        src.embed_text("DuckDB text two", "doc1.pdf", 1)

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"sqlite:///{tgt_db}",
        ):
            stats = src.migrate_backend(
                target_backend="sqlite",
                collection_name="col_a",
            )

        assert stats["success"] is True
        assert stats["migrated"] == 2
        assert stats["source_backend"] == "duckdb"
        assert stats["target_backend"] == "sqlite"

        tgt = DocEmbedder(
            "col_a", dburl=f"sqlite:///{tgt_db}", embedding_model="mxbai-embed-large"
        )
        count = tgt._count_source_records("col_a")
        assert count == 2


class TestMigrateBackendEdgeCases:
    def test_same_backend_returns_error(self, tmp_path):
        db_path = tmp_path / "source.db"
        embedder = DocEmbedder(
            "col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"sqlite:///{db_path}",
        ):
            stats = embedder.migrate_backend(target_backend="sqlite")

        assert stats["success"] is False
        assert any("same backend" in e for e in stats["errors"])

    def test_empty_source_migrates_zero(self, tmp_path):
        src_db = tmp_path / "source.db"
        tgt_db = tmp_path / "target.duckdb"

        src = DocEmbedder(
            "col", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            stats = src.migrate_backend(target_backend="duckdb")

        assert stats["success"] is True
        assert stats["total"] == 0
        assert stats["migrated"] == 0

    def test_resume_skips_existing(self, tmp_path):
        src_db = tmp_path / "source.db"
        tgt_db = tmp_path / "target.duckdb"

        src = DocEmbedder(
            "col_a", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )
        src.embed_text("First chunk", "doc1.pdf", 0)
        src.embed_text("Second chunk", "doc1.pdf", 1)

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            stats1 = src.migrate_backend(target_backend="duckdb")
            assert stats1["migrated"] == 2

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            stats2 = src.migrate_backend(
                target_backend="duckdb",
                resume=True,
            )

        assert stats2["success"] is True
        assert stats2["skipped"] == 2
        assert stats2["migrated"] == 0

    def test_batched_migration(self, tmp_path):
        src_db = tmp_path / "source.db"
        tgt_db = tmp_path / "target.duckdb"

        src = DocEmbedder(
            "col", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )
        for i in range(7):
            src.embed_text(f"Chunk number {i}", "doc1.pdf", i)

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            stats = src.migrate_backend(
                target_backend="duckdb",
                batch_size=3,
            )

        assert stats["success"] is True
        assert stats["migrated"] == 7

        tgt = DocEmbedder(
            "col", dburl=f"duckdb:///{tgt_db}", embedding_model="mxbai-embed-large"
        )
        count = tgt._count_source_records("")
        assert count == 7

    def test_dimension_mismatch_returns_error(self, tmp_path):
        src_db = tmp_path / "source.db"
        tgt_db = tmp_path / "target.duckdb"

        src = DocEmbedder(
            "col", dburl=f"sqlite:///{src_db}", embedding_model="mxbai-embed-large"
        )
        src.embed_text("Some text", "doc1.pdf", 0)

        with patch.object(
            DocEmbedder,
            "resolve_target_dburl",
            return_value=f"duckdb:///{tgt_db}",
        ):
            with patch.object(
                src, "_detect_source_dimension", return_value=512
            ):
                stats = src.migrate_backend(target_backend="duckdb")

        assert stats["success"] is False
        assert any("Dimension mismatch" in e for e in stats["errors"])
        assert stats["source_dimension"] == 512


class TestReadVectors:
    def test_read_vectors_sqlite(self, tmp_path):
        db_path = tmp_path / "source.db"
        embedder = DocEmbedder(
            "col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder.embed_text("Text A", "doc1.pdf", 0)
        embedder.embed_text("Text B", "doc2.pdf", 0)

        records = embedder._read_vectors_batch(0, 10, "")
        assert len(records) == 2
        assert records[0]["document"] == "Text A"
        assert len(records[0]["embedding"]) == 1024

    def test_read_vectors_sqlite_with_collection_filter(self, tmp_path):
        db_path = tmp_path / "source.db"
        embedder = DocEmbedder(
            "col_a", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder.embed_text("Text A", "doc1.pdf", 0)

        embedder2 = DocEmbedder(
            "col_b", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder2.embed_text("Text B", "doc2.pdf", 0)

        records = embedder._read_vectors_batch(0, 10, "col_a")
        assert len(records) == 1
        assert records[0]["collection_name"] == "col_a"

    def test_read_vectors_duckdb(self, tmp_path):
        db_path = tmp_path / "source.duckdb"
        embedder = DocEmbedder(
            "col", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder.embed_text("DuckDB text", "doc1.pdf", 0)
        embedder.embed_text("More text", "doc1.pdf", 1)

        records = embedder._read_vectors_batch(0, 10, "")
        assert len(records) == 2
        assert records[0]["embedding_model"] == "mxbai-embed-large"

    def test_read_vectors_offset_pagination(self, tmp_path):
        db_path = tmp_path / "source.db"
        embedder = DocEmbedder(
            "col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        for i in range(5):
            embedder.embed_text(f"Chunk {i}", "doc1.pdf", i)

        page1 = embedder._read_vectors_batch(0, 2, "")
        page2 = embedder._read_vectors_batch(2, 2, "")
        page3 = embedder._read_vectors_batch(4, 2, "")

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1


class TestDetectSourceDimension:
    def test_detect_sqlite_dimension(self, tmp_path):
        db_path = tmp_path / "source.db"
        embedder = DocEmbedder(
            "col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder.embed_text("Text", "doc1.pdf", 0)

        dim = embedder._detect_source_dimension()
        assert dim == 1024

    def test_detect_duckdb_dimension(self, tmp_path):
        db_path = tmp_path / "source.duckdb"
        embedder = DocEmbedder(
            "col", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder.embed_text("Text", "doc1.pdf", 0)

        dim = embedder._detect_source_dimension()
        assert dim == 1024

    def test_detect_empty_db_returns_model_dimension(self, tmp_path):
        db_path = tmp_path / "source.db"
        embedder = DocEmbedder(
            "col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        dim = embedder._detect_source_dimension()
        assert dim == 1024


class TestCountSourceRecords:
    def test_count_all_sqlite(self, tmp_path):
        db_path = tmp_path / "source.db"
        embedder = DocEmbedder(
            "col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder.embed_text("A", "doc1.pdf", 0)
        embedder.embed_text("B", "doc1.pdf", 1)
        embedder.embed_text("C", "doc2.pdf", 0)

        count = embedder._count_source_records("")
        assert count == 3

    def test_count_filtered_sqlite(self, tmp_path):
        db_path = tmp_path / "source.db"
        a = DocEmbedder(
            "col_a", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        a.embed_text("A", "doc1.pdf", 0)

        b = DocEmbedder(
            "col_b", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )
        b.embed_text("B", "doc2.pdf", 0)

        count = a._count_source_records("col_a")
        assert count == 1

    def test_count_duckdb(self, tmp_path):
        db_path = tmp_path / "source.duckdb"
        embedder = DocEmbedder(
            "col", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large"
        )
        embedder.embed_text("X", "doc1.pdf", 0)
        embedder.embed_text("Y", "doc1.pdf", 1)

        count = embedder._count_source_records("")
        assert count == 2


class TestInsertExistingVector:
    def test_insert_into_sqlite(self, tmp_path):
        db_path = tmp_path / "target.db"
        embedder = DocEmbedder(
            "col", dburl=f"sqlite:///{db_path}", embedding_model="mxbai-embed-large"
        )

        record = {
            "collection_name": "col",
            "doc_name": "doc1.pdf",
            "page_number": 0,
            "doc_hash": "abc123",
            "document": "Test document",
            "embedding_model": "mxbai-embed-large",
            "embedding": np.zeros(1024).tolist(),
        }
        embedder._insert_existing_vector(record)

        docs = embedder.get_embedded_documents()
        assert len(docs) == 1

    def test_insert_into_duckdb(self, tmp_path):
        db_path = tmp_path / "target.duckdb"
        embedder = DocEmbedder(
            "col", dburl=f"duckdb:///{db_path}", embedding_model="mxbai-embed-large"
        )

        record = {
            "collection_name": "col",
            "doc_name": "doc1.pdf",
            "page_number": 0,
            "doc_hash": "abc123",
            "document": "Test document",
            "embedding_model": "mxbai-embed-large",
            "embedding": np.zeros(1024).tolist(),
        }
        embedder._insert_existing_vector(record)

        docs = embedder.get_embedded_documents()
        assert len(docs) == 1


class TestResolveTargetDburl:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            DocEmbedder.resolve_target_dburl("cassandra")

    @patch("libbydbot.settings.Settings")
    def test_duckdb_url_format(self, mock_settings_cls, tmp_path):
        mock_settings = MagicMock()
        mock_settings.target_duckdb_path = str(tmp_path / "test.duckdb")
        mock_settings_cls.return_value = mock_settings

        url = DocEmbedder.resolve_target_dburl("duckdb")
        assert url.startswith("duckdb:///")

    @patch("libbydbot.settings.Settings")
    def test_sqlite_url_format(self, mock_settings_cls, tmp_path):
        mock_settings = MagicMock()
        mock_settings.target_sqlite_path = str(tmp_path / "test.db")
        mock_settings_cls.return_value = mock_settings

        url = DocEmbedder.resolve_target_dburl("sqlite")
        assert url.startswith("sqlite:///")

    @patch("libbydbot.settings.Settings")
    def test_postgres_unconfigured_raises(self, mock_settings_cls):
        mock_settings = MagicMock()
        mock_settings.target_postgres_url = ""
        mock_settings_cls.return_value = mock_settings

        with pytest.raises(ValueError, match="not configured"):
            DocEmbedder.resolve_target_dburl("postgresql")
