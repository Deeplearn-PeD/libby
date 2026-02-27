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
