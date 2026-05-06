import os
import shutil
import tempfile
import threading
import uuid
from hashlib import sha256
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger

from libbydbot.api.schemas import (
    EmbedJobAccepted,
    EmbedJobStatus,
    EmbedTextRequest,
    EmbedTextResponse,
    EmbedUploadResponse,
    ModelInfoResponse,
    ReembedRequest,
    ReembedResponse,
)
from libbydbot.brain.embed import DocEmbedder


router = APIRouter(prefix="/embed", tags=["embedding"])

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def get_embedder() -> DocEmbedder:
    from libbydbot.api.main import app_state

    return app_state.embedder


EmbedderDep = Annotated[DocEmbedder, Depends(get_embedder)]


def _run_embed_job(
    job_id: str,
    temp_dir: str,
    filename: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
):
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"

    try:
        from libbydbot.api.main import app_state
        embedder = app_state.embedder
        if embedder is None:
            raise RuntimeError("Embedder not initialized")

        from libbydbot.brain.ingest import PDFPipeline

        embedder.collection_name = collection_name
        chunks_embedded = 0

        pipeline = PDFPipeline(
            temp_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for text, metadata in pipeline:
            doc_name = metadata.get("title") or filename or "Unknown"
            if isinstance(text, list):
                for i, chunk in enumerate(text):
                    embedder.embed_text(chunk, doc_name, i)
                    chunks_embedded += 1
            else:
                for page_number, page_text in text.items():
                    embedder.embed_text(page_text, doc_name, page_number)
                    chunks_embedded += 1

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["chunks_embedded"] = chunks_embedded

        logger.info(f"Job {job_id} completed: {chunks_embedded} chunks embedded")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


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


@router.post("/upload", response_model=EmbedJobAccepted)
async def upload_and_embed(
    file: UploadFile = File(..., description="PDF file to upload and embed"),
    collection_name: str = Form(
        "main", description="Collection to store the document in"
    ),
    chunk_size: int = Form(800, description="Size of text chunks"),
    chunk_overlap: int = Form(100, description="Overlap between chunks"),
):
    """
    Upload a PDF file for async embedding. Returns a job ID immediately.

    Poll GET /embed/status/{job_id} for completion.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "doc_name": file.filename,
            "collection_name": collection_name,
            "chunks_embedded": 0,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_embed_job,
        args=(job_id, temp_dir, file.filename, collection_name, chunk_size, chunk_overlap),
        daemon=True,
    )
    thread.start()

    logger.info(f"Started async embed job {job_id} for '{file.filename}'")

    return EmbedJobAccepted(
        job_id=job_id,
        doc_name=file.filename,
        collection_name=collection_name,
        message="Document accepted for async embedding. Poll /embed/status/{job_id} for progress.",
    )


@router.get("/status/{job_id}", response_model=EmbedJobStatus)
def get_job_status(job_id: str):
    """
    Get the status of an async embedding job.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return EmbedJobStatus(
        job_id=job_id,
        status=job["status"],
        doc_name=job.get("doc_name", ""),
        collection_name=job.get("collection_name", ""),
        chunks_embedded=job.get("chunks_embedded", 0),
        error=job.get("error"),
    )


@router.post("/upload/sync", response_model=EmbedUploadResponse)
async def upload_and_embed_sync(
    embedder: EmbedderDep,
    file: UploadFile = File(..., description="PDF file to upload and embed"),
    collection_name: str = Form(
        "main", description="Collection to store the document in"
    ),
    chunk_size: int = Form(800, description="Size of text chunks"),
    chunk_overlap: int = Form(100, description="Overlap between chunks"),
):
    """
    Upload a PDF file and embed synchronously (blocks until done).

    Prefer /upload for large files.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from libbydbot.brain.ingest import PDFPipeline

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
    """
    try:
        embedder._migrate_add_embedding_model()

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
            backup_table=stats.get("backup_table"),
            message=f"Successfully re-embedded {stats['updated']}/{stats['total']} documents with model '{stats['new_model']}'",
        )
    except Exception as e:
        logger.error(f"Error re-embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(embedder: EmbedderDep):
    """
    Get information about embedding models used in the database.
    """
    try:
        embedder._migrate_add_embedding_model()

        info = embedder.get_embedding_model_info()

        return ModelInfoResponse(
            models=info["models"],
            total_documents=info["total_documents"],
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
