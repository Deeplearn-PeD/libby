"""
Tests for the Libby D. Bot API module.

These tests verify that the API module can be imported and the FastAPI app
can be created successfully.
"""

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
        """Test that app has expected routes."""
        from libbydbot.api.main import create_app

        test_app = create_app()
        route_paths = []
        for route in test_app.routes:
            if hasattr(route, "path"):
                route_paths.append(route.path)

        assert "/api/health" in route_paths
        assert "/api/embed/text" in route_paths
        assert "/api/retrieve" in route_paths
        assert "/api/documents" in route_paths
        assert "/api/collections" in route_paths


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
