"""
Tests for the Libby D. Bot API module.

These tests verify that the API module can be imported and the FastAPI app
can be created successfully.
"""

import pytest


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
        assert test_app.version == "0.6.0"

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

        response = HealthResponse(
            status="healthy",
            database="duckdb",
            ollama="healthy",
            version="0.6.0",
        )

        assert response.status == "healthy"
        assert response.database == "duckdb"
        assert response.ollama == "healthy"
        assert response.version == "0.6.0"

    def test_retrieved_document(self):
        """Test RetrievedDocument model."""
        from libbydbot.api.schemas import RetrievedDocument

        doc = RetrievedDocument(
            doc_name="test.pdf",
            page_number=1,
            content="Sample content",
            score=0.95,
        )

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
