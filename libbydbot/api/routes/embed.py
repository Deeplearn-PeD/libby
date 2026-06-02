import os
import shutil
import tempfile
import threading
import time
import uuid
from hashlib import sha256
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger

from libbydbot.api.schemas import (
    BackendsResponse,
    BackupTablesResponse,
    EmbedJobAccepted,
    EmbedJobListResponse,
    EmbedJobStatus,
    EmbedTextRequest,
    EmbedTextResponse,
    EmbedUploadResponse,
    EmbeddingModelsResponse,
    FinalizeRequest,
    FinalizeResponse,
    MigrateBackendRequest,
    MigrateBackendResponse,
    ModelInfoResponse,
    ReembedRequest,
    ReembedResponse,
    RollbackResponse,
    SchemaMigrationResponse,
    VerifyRequest,
    VerifyResponse,
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

        from libbydbot.brain.ingest import PDFPipeline, ChunkInfo

        embedder.collection_name = collection_name
        chunks_embedded = 0

        pipeline = PDFPipeline(
            temp_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for text, metadata in pipeline:
            doc_name = metadata.get("title") or filename or "Unknown"
            if isinstance(text, list) and text and isinstance(text[0], ChunkInfo):
                for chunk in text:
                    embedder.embed_text(chunk.text, doc_name, chunk.page_number)
                    chunks_embedded += 1
            elif isinstance(text, list):
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
            _jobs[job_id]["finished_at"] = time.time()

        logger.info(f"Job {job_id} completed: {chunks_embedded} chunks embedded")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["finished_at"] = time.time()

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
            "created_at": time.time(),
            "finished_at": None,
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
        created_at=job.get("created_at"),
        finished_at=job.get("finished_at"),
    )


@router.get("/jobs", response_model=EmbedJobListResponse)
def list_jobs():
    """
    List all embed jobs with their statuses.

    Completed and failed jobs older than 5 minutes are automatically pruned.
    """
    now = time.time()
    stale_threshold = 300

    with _jobs_lock:
        stale_ids = [
            jid for jid, j in _jobs.items()
            if j.get("finished_at") and (now - j["finished_at"]) > stale_threshold
        ]
        for jid in stale_ids:
            del _jobs[jid]

        jobs = [
            EmbedJobStatus(
                job_id=jid,
                status=j["status"],
                doc_name=j.get("doc_name", ""),
                collection_name=j.get("collection_name", ""),
                chunks_embedded=j.get("chunks_embedded", 0),
                error=j.get("error"),
                created_at=j.get("created_at"),
                finished_at=j.get("finished_at"),
            )
            for jid, j in _jobs.items()
        ]

    processing = sum(1 for j in jobs if j.status == "processing")
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")

    return EmbedJobListResponse(
        jobs=jobs,
        processing=processing,
        completed=completed,
        failed=failed,
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

        from libbydbot.brain.ingest import PDFPipeline, ChunkInfo

        embedder.collection_name = collection_name
        chunks_embedded = 0

        pipeline = PDFPipeline(
            temp_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for text, metadata in pipeline:
            doc_name = metadata.get("title") or file.filename or "Unknown"
            if isinstance(text, list) and text and isinstance(text[0], ChunkInfo):
                for chunk in text:
                    embedder.embed_text(chunk.text, doc_name, chunk.page_number)
                    chunks_embedded += 1
            elif isinstance(text, list):
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
            rechunk=request.rechunk,
            new_chunk_size=request.new_chunk_size,
            new_chunk_overlap=request.new_chunk_overlap,
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
            total_old_chunks=stats.get("total_old_chunks", 0),
            total_new_chunks=stats.get("total_new_chunks", 0),
            old_chunk_size=stats.get("old_chunk_size", 0),
            new_chunk_size=stats.get("new_chunk_size", 0),
            shadow_collection=stats.get("shadow_collection", ""),
            shadow_table=stats.get("shadow_table", ""),
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


@router.get("/backends", response_model=BackendsResponse)
def list_backends(embedder: EmbedderDep):
    """List available database backends and their configuration status."""
    try:
        backends = embedder.list_backends()
        current = next(
            (b["name"] for b in backends if b["is_current"]), "unknown"
        )
        return BackendsResponse(backends=backends, current=current)
    except Exception as e:
        logger.error(f"Error listing backends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=EmbeddingModelsResponse)
def list_embedding_models():
    """List available embedding models and their configuration."""
    from libbydbot.settings import Settings

    settings = Settings()
    models = [
        {
            "name": name,
            "code": details["code"],
            "is_default": details.get("is_default", False),
        }
        for name, details in settings.embedding_models.items()
    ]
    default_code = settings.default_embedding_model
    return EmbeddingModelsResponse(models=models, default=default_code)


@router.get("/backups", response_model=BackupTablesResponse)
def list_backup_tables(embedder: EmbedderDep):
    """List all backup embedding tables found in the database."""
    try:
        backups_raw = embedder.list_backup_tables()
        current_count = 0
        try:
            if embedder.dburl.startswith(("duckdb", "sqlite")):
                current_count = embedder.connection.execute(
                    f"SELECT COUNT(*) FROM {embedder.table_name}"
                ).fetchone()[0]
            else:
                from sqlalchemy import text as sa_text
                with embedder.engine.connect() as conn:
                    current_count = conn.execute(
                        sa_text(f"SELECT COUNT(*) FROM {embedder.table_name}")
                    ).scalar() or 0
        except Exception:
            pass
        return BackupTablesResponse(
            backups=backups_raw,
            current_table=embedder.table_name,
            current_count=current_count,
            current_model=embedder.embedding_model,
            total_backups=len(backups_raw),
        )
    except Exception as e:
        logger.error(f"Error listing backup tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback", response_model=RollbackResponse)
def rollback_embedding(embedder: EmbedderDep, dry_run: bool = False):
    """
    Rollback to the backup table created by a previous re-embed operation.

    Only supported for DuckDB, where the backup table is preserved. Use
    ``dry_run=true`` to preview without making changes.
    """
    try:
        result = embedder.rollback_embedding(dry_run=dry_run)

        return RollbackResponse(
            success=result["success"],
            message=result["message"],
            backend=result["backend"],
            backup_table=result.get("backup_table"),
            current_model=result.get("current_model", ""),
            backup_model=result.get("backup_model", ""),
            backup_count=result.get("backup_count", 0),
            current_count=result.get("current_count", 0),
            restored_count=result.get("restored_count", 0),
            dry_run=result.get("dry_run", dry_run),
        )
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/migrate", response_model=MigrateBackendResponse)
def migrate_backend(request: MigrateBackendRequest, embedder: EmbedderDep):
    """Migrate embeddings to a different database backend."""
    try:
        stats = embedder.migrate_backend(
            target_backend=request.target_backend,
            collection_name=request.collection_name,
            batch_size=request.batch_size,
            dry_run=request.dry_run,
            resume=request.resume,
        )
        return MigrateBackendResponse(
            success=stats["success"],
            total=stats["total"],
            migrated=stats["migrated"],
            skipped=stats["skipped"],
            errors=stats["errors"],
            source_backend=stats["source_backend"],
            target_backend=stats["target_backend"],
            source_dimension=stats["source_dimension"],
            target_dimension=stats["target_dimension"],
            message="Migration complete" if stats["success"] else "Migration failed",
        )
    except Exception as e:
        logger.error(f"Error migrating backend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finalize", response_model=FinalizeResponse)
def finalize_rechunk(request: FinalizeRequest, embedder: EmbedderDep):
    """
    Finalize a rechunk operation: cutover from shadow collection to production.

    After a ``reembed`` with ``rechunk=true``, the new data lives in a shadow
    collection (``{name}_v2``).  This endpoint swaps it into place.
    """
    try:
        stats = embedder.finalize_rechunk(
            collection_name=request.collection_name,
            shadow_collection=request.shadow_collection,
            shadow_table=request.shadow_table,
        )
        return FinalizeResponse(
            success=stats["success"],
            collection=stats["collection"],
            original_count=stats["original_count"],
            shadow_count=stats["shadow_count"],
            deleted_original=stats["deleted_original"],
            renamed=stats["renamed"],
            table_swapped=stats["table_swapped"],
            errors=stats["errors"],
            message="Finalized successfully" if stats["success"] else f"Finalize failed: {'; '.join(stats['errors'])}",
        )
    except Exception as e:
        logger.error(f"Error finalizing rechunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/migrate-schema", response_model=SchemaMigrationResponse)
def migrate_schema(embedder: EmbedderDep):
    """
    Migrate embedding tables to use compound UNIQUE(collection_name, doc_hash)
    instead of global UNIQUE(doc_hash).

    This allows the same text to appear in multiple collections.
    """
    try:
        result = embedder.migrate_compound_unique()
        return SchemaMigrationResponse(
            backend=result["backend"],
            changes=result["changes"],
        )
    except Exception as e:
        logger.error(f"Error migrating schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify", response_model=VerifyResponse)
def verify_embeddings(request: VerifyRequest, embedder: EmbedderDep):
    """
    Verify embedding data integrity and optionally fix issues.

    Checks include: duplicate hashes, hash integrity, missing models,
    mixed models, dimension consistency, partial documents, orphaned
    shadow collections, orphaned tables, stale backups, empty embeddings,
    and duplicate doc+page combinations.
    """
    try:
        result = embedder.verify_and_fix(
            collection_name=request.collection_name,
            dry_run=request.dry_run,
            checks=request.checks,
            auto_finalize=request.auto_finalize,
        )
        return VerifyResponse(
            collection=result["collection"],
            dry_run=result["dry_run"],
            table=result["table"],
            checks=result["checks"],
            summary=result["summary"],
            errors=result.get("errors", []),
            finalized=result.get("finalized", []),
        )
    except Exception as e:
        logger.error(f"Error verifying embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
