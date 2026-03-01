import os
import tempfile
import shutil
from hashlib import sha256
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger

from libbydbot.api.schemas import (
    EmbedTextRequest,
    EmbedTextResponse,
    EmbedUploadResponse,
    ReembedRequest,
    ReembedResponse,
    ModelInfoResponse,
)
from libbydbot.brain.embed import DocEmbedder


router = APIRouter(prefix="/embed", tags=["embedding"])


def get_embedder() -> DocEmbedder:
    from libbydbot.api.main import app_state

    return app_state.embedder


EmbedderDep = Annotated[DocEmbedder, Depends(get_embedder)]


@router.post("/text", response_model=EmbedTextResponse)
def embed_text(request: EmbedTextRequest, embedder: EmbedderDep):
    """
    Embed raw text into the document database.

    - **text**: The text content to embed
    - **doc_name**: Name/identifier for the document
    - **page_number**: Page number or chunk index (default: 0)
    - **collection_name**: Collection to store the document in (default: "main")
    """
    try:
        doc_hash = sha256(request.text.encode()).hexdigest()

        embedder.collection_name = request.collection_name
        embedder.embed_text(request.text, request.doc_name, request.page_number)

        return EmbedTextResponse(
            success=True,
            doc_name=request.doc_name,
            doc_hash=doc_hash,
            message=f"Successfully embedded text from '{request.doc_name}' into collection '{request.collection_name}'",
        )
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=EmbedUploadResponse)
async def upload_and_embed(
    embedder: EmbedderDep,
    file: UploadFile = File(..., description="PDF file to upload and embed"),
    collection_name: str = Form(
        "main", description="Collection to store the document in"
    ),
    chunk_size: int = Form(800, description="Size of text chunks"),
    chunk_overlap: int = Form(100, description="Overlap between chunks"),
):
    """
    Upload a PDF file and embed its contents.

    - **file**: PDF file to upload
    - **collection_name**: Collection to store the document in (default: "main")
    - **chunk_size**: Size of text chunks for embedding (default: 800)
    - **chunk_overlap**: Overlap between consecutive chunks (default: 100)
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from libbydbot.brain.ingest import PDFPipeline, TextSplitter

        embedder.collection_name = collection_name
        chunks_embedded = 0

        pipeline = PDFPipeline(
            temp_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for text, metadata in pipeline:
            doc_name = metadata.get("title") or file.filename or "Unknown"
            if isinstance(text, list):
                for i, chunk in enumerate(text):
                    embedder.embed_text(chunk, doc_name, i)
                    chunks_embedded += 1
            else:
                for page_number, page_text in text.items():
                    embedder.embed_text(page_text, doc_name, page_number)
                    chunks_embedded += 1

        return EmbedUploadResponse(
            success=True,
            doc_name=file.filename,
            chunks_embedded=chunks_embedded,
            collection_name=collection_name,
            message=f"Successfully embedded {chunks_embedded} chunks from '{file.filename}' into collection '{collection_name}'",
        )
    except Exception as e:
        logger.error(f"Error uploading and embedding file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/reembed", response_model=ReembedResponse)
def reembed_embeddings(request: ReembedRequest, embedder: EmbedderDep):
    """
    Re-embed documents with a new embedding model.

    This operation:
    1. Migrates the database schema if needed (adds embedding_model column)
    2. Regenerates embeddings for all matching documents
    3. Updates the embedding_model field for each document

    - **collection_name**: Collection to re-embed (empty for all)
    - **new_model**: New embedding model (empty for settings default)
    - **batch_size**: Batch size for processing (default: 100)
    """
    try:
        # Run migration first
        embedder._migrate_add_embedding_model()

        # Perform re-embedding
        stats = embedder.reembed(
            collection_name=request.collection_name,
            new_model=request.new_model if request.new_model else None,
            batch_size=request.batch_size,
        )

        return ReembedResponse(
            success=True,
            total=stats["total"],
            updated=stats["updated"],
            old_model=stats["old_model"],
            new_model=stats["new_model"],
            errors=stats["errors"],
            message=f"Successfully re-embedded {stats['updated']}/{stats['total']} documents with model '{stats['new_model']}'",
        )
    except Exception as e:
        logger.error(f"Error re-embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(embedder: EmbedderDep):
    """
    Get information about embedding models used in the database.

    Returns a breakdown of documents by model and collection.
    """
    try:
        # Run migration first to ensure column exists
        embedder._migrate_add_embedding_model()

        info = embedder.get_embedding_model_info()

        return ModelInfoResponse(
            models=info["models"],
            total_documents=info["total_documents"],
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
