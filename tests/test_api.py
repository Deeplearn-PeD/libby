"""
Tests for the Libby D. Bot API module.

These tests verify that the API module can be imported and the FastAPI app
can be created successfully.
"""

import os
from pathlib import Path

import pytest

from libbydbot.brain.embed import DocEmbedder


class TestAPIImports:
    """Test that API modules can be imported correctly."""

    def test_import_schemas(self):
        """Test that schemas module can be imported."""
        from libbydbot.api.schemas import (
            EmbedTextRequest,
            EmbedTextResponse,
            EmbedUploadResponse,
            RetrievedDocument,
            RetrieveRequest,
            RetrieveResponse,
            DocumentInfo,
            DocumentListResponse,
            CollectionInfo,
            CollectionListResponse,
            HealthResponse,
            ErrorResponse,
        )

        assert EmbedTextRequest is not None
        assert RetrieveRequest is not None
        assert HealthResponse is not None

    def test_import_routes_embed(self):
        """Test that embed routes module can be imported."""
        from libbydbot.api.routes import embed

        assert embed.router is not None
        assert embed.router.prefix == "/embed"

    def test_import_routes_retrieve(self):
        """Test that retrieve routes module can be imported."""
        from libbydbot.api.routes import retrieve

        assert retrieve.router is not None
        assert retrieve.router.prefix == "/api"

    def test_import_main(self):
        """Test that main module can be imported."""
        from libbydbot.api.main import app, create_app, run

        assert app is not None
        assert create_app is not None
        assert run is not None


class TestAppCreation:
    """Test FastAPI app creation."""

    def test_create_app(self):
        """Test that FastAPI app can be created."""
        from libbydbot.api.main import create_app

        test_app = create_app()
        assert test_app.title == "Libby D. Bot API"
        from libbydbot.api.main import VERSION
        assert test_app.version == VERSION

    def test_app_routes(self):
        """Test that app has expected routes.

        FastAPI >= 0.115 includes routers lazily, so routes added via
        include_router do not appear directly in app.routes. Inspecting the
        generated OpenAPI schema resolves all routers reliably.
        """
        from libbydbot.api.main import create_app

        test_app = create_app()
        route_paths = set(test_app.openapi()["paths"].keys())

        assert "/api/health" in route_paths
        assert "/api/embed/text" in route_paths
        assert "/api/retrieve" in route_paths
        assert "/api/documents" in route_paths
        assert "/api/collections" in route_paths
        # Wiki browsing endpoints
        assert "/api/wiki/pages/{collection_name}" in route_paths
        assert "/api/wiki/page/{collection_name}" in route_paths


class TestSchemaModels:
    """Test Pydantic schema models."""

    def test_embed_text_request(self):
        """Test EmbedTextRequest model."""
        from libbydbot.api.schemas import EmbedTextRequest

        request = EmbedTextRequest(
            text="Sample text",
            doc_name="test.txt",
        )

        assert request.text == "Sample text"
        assert request.doc_name == "test.txt"
        assert request.page_number == 0
        assert request.collection_name == "main"

    def test_embed_text_request_custom(self):
        """Test EmbedTextRequest with custom values."""
        from libbydbot.api.schemas import EmbedTextRequest

        request = EmbedTextRequest(
            text="Sample text",
            doc_name="test.txt",
            page_number=5,
            collection_name="custom",
        )

        assert request.page_number == 5
        assert request.collection_name == "custom"

    def test_retrieve_request(self):
        """Test RetrieveRequest model."""
        from libbydbot.api.schemas import RetrieveRequest

        request = RetrieveRequest(query="What is AI?")

        assert request.query == "What is AI?"
        assert request.collection_name == ""
        assert request.num_docs == 5

    def test_retrieve_request_custom(self):
        """Test RetrieveRequest with custom values."""
        from libbydbot.api.schemas import RetrieveRequest

        request = RetrieveRequest(
            query="What is AI?",
            collection_name="main",
            num_docs=10,
        )

        assert request.collection_name == "main"
        assert request.num_docs == 10

    def test_health_response(self):
        """Test HealthResponse model."""
        from libbydbot.api.schemas import HealthResponse

        from libbydbot.api.main import VERSION

        response = HealthResponse(
            status="healthy",
            database="duckdb",
            ollama="healthy",
            version=VERSION,
        )

        assert response.status == "healthy"
        assert response.database == "duckdb"
        assert response.ollama == "healthy"
        assert response.version == VERSION

    def test_retrieved_document(self):
        """Test RetrievedDocument model."""
        from libbydbot.api.schemas import RetrievedDocument

        doc = RetrievedDocument(
            collection_name="test_collection",
            doc_name="test.pdf",
            page_number=1,
            content="Sample content",
            score=0.95,
        )

        assert doc.collection_name == "test_collection"
        assert doc.doc_name == "test.pdf"
        assert doc.page_number == 1
        assert doc.content == "Sample content"
        assert doc.score == 0.95

    def test_document_info(self):
        """Test DocumentInfo model."""
        from libbydbot.api.schemas import DocumentInfo

        doc = DocumentInfo(
            doc_name="test.pdf",
            collection_name="main",
        )

        assert doc.doc_name == "test.pdf"
        assert doc.collection_name == "main"

    def test_collection_info(self):
        """Test CollectionInfo model."""
        from libbydbot.api.schemas import CollectionInfo

        collection = CollectionInfo(
            name="main",
            document_count=10,
        )

        assert collection.name == "main"
        assert collection.document_count == 10

    def test_document_list_response(self):
        """Test DocumentListResponse model."""
        from libbydbot.api.schemas import DocumentListResponse, DocumentInfo

        response = DocumentListResponse(
            documents=[
                DocumentInfo(doc_name="doc1.pdf", collection_name="main"),
                DocumentInfo(doc_name="doc2.pdf", collection_name="main"),
            ],
            total=2,
        )

        assert len(response.documents) == 2
        assert response.total == 2

    def test_collection_list_response(self):
        """Test CollectionListResponse model."""
        from libbydbot.api.schemas import CollectionListResponse, CollectionInfo

        response = CollectionListResponse(
            collections=[
                CollectionInfo(name="main", document_count=10),
                CollectionInfo(name="secondary", document_count=5),
            ],
            total=2,
        )

        assert len(response.collections) == 2
        assert response.total == 2

    def test_retrieve_response(self):
        """Test RetrieveResponse model."""
        from libbydbot.api.schemas import (
            RetrieveResponse,
            RetrievedDocument,
        )

        response = RetrieveResponse(
            query="test query",
            collection_name="main",
            documents=[
                RetrievedDocument(
                    collection_name="main",
                    doc_name="test.pdf",
                    page_number=1,
                    content="content",
                    score=0.9,
                ),
            ],
            total=1,
        )

        assert response.query == "test query"
        assert response.collection_name == "main"
        assert len(response.documents) == 1
        assert response.total == 1

    def test_embed_text_response(self):
        """Test EmbedTextResponse model."""
        from libbydbot.api.schemas import EmbedTextResponse

        response = EmbedTextResponse(
            success=True,
            doc_name="test.txt",
            doc_hash="abc123",
            message="Successfully embedded",
        )

        assert response.success is True
        assert response.doc_name == "test.txt"
        assert response.doc_hash == "abc123"
        assert response.message == "Successfully embedded"

    def test_embed_upload_response(self):
        """Test EmbedUploadResponse model."""
        from libbydbot.api.schemas import EmbedUploadResponse

        response = EmbedUploadResponse(
            success=True,
            doc_name="test.pdf",
            chunks_embedded=10,
            collection_name="main",
            message="Successfully embedded 10 chunks",
        )

        assert response.success is True
        assert response.doc_name == "test.pdf"
        assert response.chunks_embedded == 10
        assert response.collection_name == "main"

    def test_error_response(self):
        """Test ErrorResponse model."""
        from libbydbot.api.schemas import ErrorResponse

        response = ErrorResponse(
            error="ValueError",
            message="Invalid input",
            detail="Text cannot be empty",
        )

        assert response.error == "ValueError"
        assert response.message == "Invalid input"
        assert response.detail == "Text cannot be empty"


class TestUploadEmbed:
    """Test PDF upload and embedding via the API."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a TestClient with a fresh SQLite-backed embedder."""
        from fastapi.testclient import TestClient
        from libbydbot.api.main import app_state, create_app

        db_path = tmp_path / "test_upload.db"
        embedder = DocEmbedder(
            "test_upload",
            dburl=f"sqlite:///{db_path}",
            embedding_model="mxbai-embed-large",
        )

        app_state.embedder = embedder
        test_app = create_app()
        client = TestClient(test_app)
        yield client

        app_state.embedder = None

    def test_upload_sync_embeds_pdf(self, client):
        """Upload a real PDF via /api/embed/upload/sync and verify chunks are stored."""
        corpus = Path(__file__).parent / "test_corpus"
        pdf_files = sorted(corpus.glob("*.pdf"))
        assert len(pdf_files) > 0, f"No PDFs found in {corpus}"

        pdf_path = pdf_files[0]

        with open(pdf_path, "rb") as f:
            response = client.post(
                "/api/embed/upload/sync",
                files={"file": (pdf_path.name, f, "application/pdf")},
                data={
                    "collection_name": "test_upload",
                    "chunk_size": "800",
                    "chunk_overlap": "100",
                },
            )

        assert response.status_code == 200, f"Upload failed: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert data["chunks_embedded"] > 0
        assert data["collection_name"] == "test_upload"

        docs = client.get("/api/documents", params={"collection_name": "test_upload"})
        assert docs.status_code == 200
        doc_list = docs.json()
        assert len(doc_list["documents"]) > 0

    def test_upload_multiple_pdfs(self, client):
        """Upload all PDFs from test_corpus and verify each is embedded."""
        corpus = Path(__file__).parent / "test_corpus"
        pdf_files = sorted(corpus.glob("*.pdf"))

        total_chunks = 0
        for pdf_path in pdf_files:
            with open(pdf_path, "rb") as f:
                response = client.post(
                    "/api/embed/upload/sync",
                    files={"file": (pdf_path.name, f, "application/pdf")},
                    data={
                        "collection_name": "multi_upload",
                        "chunk_size": "800",
                        "chunk_overlap": "100",
                    },
                )
            assert response.status_code == 200, f"Upload {pdf_path.name} failed: {response.text}"
            total_chunks += response.json()["chunks_embedded"]

        assert total_chunks > 0

        docs = client.get("/api/documents", params={"collection_name": "multi_upload"})
        doc_names = {d["doc_name"] for d in docs.json()["documents"]}
        assert len(doc_names) == len(pdf_files)


class TestWikiBrowse:
    """Test the wiki browsing endpoints (list pages + read a page)."""

    @pytest.fixture
    def wiki_client(self, tmp_path, monkeypatch):
        """Create a TestClient backed by a temp wiki with sample pages."""
        os.environ["EMBED_DB"] = f"sqlite:///{tmp_path / 'embed.db'}"

        from fastapi.testclient import TestClient
        from libbydbot.api.main import create_app
        from libbydbot.api.routes import wiki as wiki_routes
        from libbydbot.brain.wiki import WikiManager

        wiki = WikiManager(
            collection_name="main", wiki_base=str(tmp_path), model="llama3.2"
        )
        (wiki.sources_dir / "doc1.md").write_text(
            "---\ntitle: Doc1\n---\n\n# Doc1\n\nHello world.", encoding="utf-8"
        )
        (wiki.entities_dir / "person_a.md").write_text(
            "---\ntitle: Person A\n---\n\n# Person A", encoding="utf-8"
        )

        def fake_get_wiki_manager(collection_name: str = "main") -> WikiManager:
            return WikiManager(
                collection_name=collection_name,
                wiki_base=str(tmp_path),
                model="llama3.2",
            )

        monkeypatch.setattr(wiki_routes, "get_wiki_manager", fake_get_wiki_manager)

        client = TestClient(create_app())
        yield client
        os.environ.pop("EMBED_DB", None)

    def test_browse_returns_categories(self, wiki_client):
        """GET /api/wiki/pages/main lists pages grouped by category."""
        response = wiki_client.get("/api/wiki/pages/main")
        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "main"
        assert set(data["categories"].keys()) == {
            "sources",
            "entities",
            "concepts",
            "synthesis",
        }
        assert data["categories"]["sources"] == ["doc1"]
        assert data["categories"]["entities"] == ["person_a"]
        assert "index" in data["root_pages"]
        assert "log" in data["root_pages"]

    def test_read_source_page(self, wiki_client):
        """GET /api/wiki/page/main reads a source page."""
        response = wiki_client.get(
            "/api/wiki/page/main", params={"category": "sources", "page": "doc1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "sources"
        assert data["page"] == "doc1"
        assert "# Doc1" in data["content"]
        assert data["path"].endswith("doc1.md")

    def test_read_root_page(self, wiki_client):
        """GET /api/wiki/page/main reads the root index page."""
        response = wiki_client.get(
            "/api/wiki/page/main", params={"category": "root", "page": "index"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "root"
        assert "Wiki Index" in data["content"]

    def test_missing_page_returns_404(self, wiki_client):
        """A non-existent page returns 404."""
        response = wiki_client.get(
            "/api/wiki/page/main", params={"category": "sources", "page": "nope"}
        )
        assert response.status_code == 404

    def test_invalid_category_returns_400(self, wiki_client):
        """An unknown category returns 400."""
        response = wiki_client.get(
            "/api/wiki/page/main", params={"category": "evil", "page": "x"}
        )
        assert response.status_code == 400

    def test_path_traversal_blocked(self, wiki_client):
        """A page name with path separators is rejected."""
        response = wiki_client.get(
            "/api/wiki/page/main",
            params={"category": "sources", "page": "../index"},
        )
        assert response.status_code == 400


class TestWikiIngestFromEmbeddings:
    """Test the DB-driven wiki ingest endpoint and auto-ingest wiring."""

    @pytest.fixture
    def ingest_client(self, tmp_path, monkeypatch):
        """A TestClient with a real SQLite embedder + wiki pointing at temp dirs."""
        import numpy as np
        from unittest.mock import patch

        os.environ["EMBED_DB"] = f"sqlite:///{tmp_path / 'embed.db'}"
        os.environ["WIKI_BASE_PATH"] = str(tmp_path / "wikis")

        from fastapi.testclient import TestClient
        from libbydbot.api.main import app_state, create_app
        from libbydbot.brain.embed import DocEmbedder

        embed_patch = patch("libbydbot.brain.embed.DocEmbedder._generate_embedding")
        dim_patch = patch(
            "libbydbot.brain.embed.DocEmbedder._get_embedding_dimension",
            return_value=1024,
        )
        mocked = embed_patch.start()
        dim_patch.start()
        mocked.return_value = np.zeros(1024).tolist()

        embedder = DocEmbedder(
            "main",
            dburl=f"sqlite:///{tmp_path / 'embed.db'}",
            embedding_model="mxbai-embed-large",
        )
        embedder.embed_text("Content about Carnegie and steel.", "doc_one", 0)
        app_state.embedder = embedder

        client = TestClient(create_app())
        yield client

        embed_patch.stop()
        dim_patch.stop()
        app_state.embedder = None
        os.environ.pop("EMBED_DB", None)
        os.environ.pop("WIKI_BASE_PATH", None)

    def test_ingest_from_embeddings_endpoint(self, ingest_client, monkeypatch):
        """POST /api/wiki/ingest-from-embeddings builds the wiki from the DB."""
        from libbydbot.brain.wiki import WikiManager

        canned = {
            "collection": "main",
            "documents_ingested": 1,
            "pages_touched": 3,
            "results": [
                {
                    "source": "doc_one",
                    "pages_touched": 3,
                    "entities_created": 1,
                    "concepts_created": 1,
                    "summary": "A summary.",
                }
            ],
        }
        monkeypatch.setattr(
            WikiManager, "ingest_from_embeddings", lambda self, *a, **k: canned
        )

        response = ingest_client.post(
            "/api/wiki/ingest-from-embeddings",
            json={"collection_name": "main", "doc_name": "doc_one"},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["documents_ingested"] == 1
        assert data["pages_touched"] >= 1
        assert data["results"][0]["source"] == "doc_one"

    def test_ingest_from_embeddings_route_registered(self):
        """The new endpoint is present in the OpenAPI schema."""
        from libbydbot.api.main import create_app

        paths = set(create_app().openapi()["paths"].keys())
        assert "/api/wiki/ingest-from-embeddings" in paths

    def test_auto_ingest_helper_exists(self):
        """The embed routes expose the auto-ingest helper."""
        from libbydbot.api.routes import embed

        assert callable(getattr(embed, "_maybe_auto_ingest_wiki", None))

    def test_auto_ingest_disabled_when_setting_off(self, tmp_path, monkeypatch):
        """Auto-ingest is a no-op (and never raises) when the setting is off."""
        from libbydbot.api.routes.embed import _maybe_auto_ingest_wiki
        from libbydbot.brain.embed import DocEmbedder

        monkeypatch.setenv("WIKI_AUTO_INGEST", "false")
        embedder = DocEmbedder("main", dburl=f"sqlite:///{tmp_path}/e.db")
        # Should return None without error even with a bare embedder.
        assert _maybe_auto_ingest_wiki(embedder, "main", "x") is None


class TestWikiConsolidate:
    """Test the POST /api/wiki/consolidate endpoint."""

    @pytest.fixture
    def consolidate_client(self, tmp_path, monkeypatch):
        os.environ["EMBED_DB"] = f"sqlite:///{tmp_path / 'embed.db'}"

        from fastapi.testclient import TestClient
        from libbydbot.api.main import create_app
        from libbydbot.api.routes import wiki as wiki_routes
        from libbydbot.brain.wiki import WikiManager

        wiki = WikiManager(
            collection_name="main", wiki_base=str(tmp_path), model="llama3.2"
        )
        for stem, body in (
            ("report_part1.md", "# Report Part 1\n\nFirst part.\n"),
            ("report_part2.md", "# Report Part 2\n\nSecond part.\n"),
            ("standalone.md", "# Standalone\n\nUnrelated.\n"),
        ):
            (wiki.sources_dir / stem).write_text(body, encoding="utf-8")

        def fake_get_wiki_manager(collection_name: str = "main") -> WikiManager:
            return WikiManager(
                collection_name=collection_name,
                wiki_base=str(tmp_path),
                model="llama3.2",
            )

        monkeypatch.setattr(wiki_routes, "get_wiki_manager", fake_get_wiki_manager)

        client = TestClient(create_app())
        yield client
        os.environ.pop("EMBED_DB", None)

    def test_consolidate_endpoint_merges_parts(self, consolidate_client):
        response = consolidate_client.post(
            "/api/wiki/consolidate", json={"collection_name": "main"}
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["groups_merged"] == 1
        assert data["pages_removed"] == 2

    def test_consolidate_route_registered(self):
        from libbydbot.api.main import create_app

        paths = set(create_app().openapi()["paths"].keys())
        assert "/api/wiki/consolidate" in paths
