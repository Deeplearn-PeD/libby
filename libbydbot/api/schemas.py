from pydantic import BaseModel, Field


class EmbedTextRequest(BaseModel):
    text: str = Field(..., description="Text content to embed")
    doc_name: str = Field(..., description="Name of the document")
    page_number: int = Field(0, description="Page number or chunk index")
    collection_name: str = Field(
        "main", description="Collection to store the document in"
    )


class EmbedTextResponse(BaseModel):
    success: bool = Field(..., description="Whether the embedding was successful")
    doc_name: str = Field(..., description="Name of the embedded document")
    doc_hash: str | None = Field(None, description="Hash of the document content")
    message: str = Field(..., description="Status message")


class EmbedUploadResponse(BaseModel):
    success: bool = Field(
        ..., description="Whether the upload and embedding was successful"
    )
    doc_name: str = Field(..., description="Name of the uploaded document")
    chunks_embedded: int = Field(..., description="Number of chunks embedded")
    collection_name: str = Field(
        ..., description="Collection the document was stored in"
    )
    message: str = Field(..., description="Status message")


class RetrievedDocument(BaseModel):
    doc_name: str = Field(..., description="Name of the document")
    page_number: int = Field(..., description="Page number or chunk index")
    content: str = Field(..., description="Document content")
    collection_name: str = Field(..., description="Collection the document belongs to")
    score: float = Field(..., description="Relevance score (higher is better)")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    collection_name: str = Field(
        "", description="Collection to search in (empty for all)"
    )
    num_docs: int = Field(
        5, ge=1, le=100, description="Number of documents to retrieve"
    )


class RetrieveResponse(BaseModel):
    query: str = Field(..., description="The search query")
    collection_name: str = Field(..., description="Collection searched")
    documents: list[RetrievedDocument] = Field(..., description="Retrieved documents")
    total: int = Field(..., description="Total number of documents retrieved")


class DocumentInfo(BaseModel):
    doc_name: str = Field(..., description="Name of the document")
    collection_name: str = Field(..., description="Collection the document belongs to")


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")


class CollectionInfo(BaseModel):
    name: str = Field(..., description="Name of the collection")
    document_count: int = Field(
        ..., description="Number of documents in the collection"
    )


class CollectionListResponse(BaseModel):
    collections: list[CollectionInfo] = Field(..., description="List of collections")
    total: int = Field(..., description="Total number of collections")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    database: str = Field(..., description="Database type being used")
    ollama: str = Field(..., description="Ollama service status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional details")


class ReembedRequest(BaseModel):
    collection_name: str = Field(
        "", description="Collection to re-embed (empty for all)"
    )
    new_model: str = Field(
        "", description="New embedding model (empty for settings default)"
    )
    batch_size: int = Field(100, ge=1, le=1000, description="Batch size for processing")
    rechunk: bool = Field(True, description="Reconstruct source text and re-chunk before embedding")
    new_chunk_size: int = Field(1500, ge=100, le=8000, description="Chunk size when rechunking")
    new_chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap when rechunking")


class ReembedResponse(BaseModel):
    success: bool = Field(..., description="Whether the re-embedding was successful")
    total: int = Field(..., description="Total documents to re-embed")
    updated: int = Field(..., description="Number of documents successfully updated")
    old_model: str = Field(..., description="Previous embedding model")
    new_model: str = Field(..., description="New embedding model used")
    errors: list[str] = Field(
        default_factory=list, description="List of errors encountered"
    )
    backup_table: str | None = Field(
        None, description="Name of backup table (for DuckDB, kept for safety)"
    )
    message: str = Field(..., description="Status message")
    total_old_chunks: int = Field(0, description="Original chunk count (when rechunking)")
    total_new_chunks: int = Field(0, description="New chunk count (when rechunking)")
    old_chunk_size: int = Field(0, description="Previous chunk size (when rechunking)")
    new_chunk_size: int = Field(0, description="New chunk size (when rechunking)")
    shadow_collection: str = Field("", description="Name of shadow collection (when rechunking)")
    shadow_table: str = Field("", description="Dimension-specific table used for shadow (when rechunking)")


class ModelInfoResponse(BaseModel):
    models: dict[str, dict[str, int]] = Field(
        ..., description="Model usage by collection (model -> collection -> count)"
    )
    total_documents: int = Field(..., description="Total number of documents")


class BackendInfo(BaseModel):
    name: str = Field(..., description="Backend identifier: postgresql, duckdb, sqlite")
    display_name: str = Field(..., description="Human-readable backend name")
    is_current: bool = Field(..., description="Whether this is the active backend")
    is_configured: bool = Field(..., description="Whether the backend URL/path is configured")
    location: str = Field("", description="Safe display location (no credentials)")


class BackendsResponse(BaseModel):
    backends: list[BackendInfo] = Field(..., description="Available database backends")
    current: str = Field(..., description="Name of the current active backend")


class MigrateBackendRequest(BaseModel):
    target_backend: str = Field(
        ..., description="Target backend: postgresql, duckdb, or sqlite"
    )
    collection_name: str = Field(
        "", description="Collection to migrate (empty for all)"
    )
    batch_size: int = Field(1000, ge=1, le=10000, description="Batch size for processing")
    dry_run: bool = Field(False, description="Preview without making changes")
    resume: bool = Field(False, description="Skip already-migrated records")


class MigrateBackendResponse(BaseModel):
    success: bool = Field(..., description="Whether the migration was successful")
    total: int = Field(..., description="Total records to migrate")
    migrated: int = Field(..., description="Number of records migrated")
    skipped: int = Field(..., description="Number of records skipped (resume mode)")
    errors: list[str] = Field(default_factory=list, description="List of errors encountered")
    source_backend: str = Field(..., description="Source backend name")
    target_backend: str = Field(..., description="Target backend name")
    source_dimension: int = Field(..., description="Source embedding dimension")
    target_dimension: int = Field(..., description="Target embedding dimension")
    message: str = Field(..., description="Status message")


# ── Wiki schemas ──


class WikiIngestRequest(BaseModel):
    doc_name: str = Field(..., description="Name of the document to ingest")
    doc_content: str = Field(..., description="Full text content of the document")
    collection_name: str = Field(
        "main", description="Collection whose wiki to update"
    )
    source_type: str = Field("document", description="Type of source")


class WikiIngestResponse(BaseModel):
    success: bool = Field(..., description="Whether ingest succeeded")
    source: str = Field(..., description="Source document name")
    pages_touched: int = Field(..., description="Number of wiki pages created/updated")
    entities_created: int = Field(..., description="Number of entity pages created/updated")
    concepts_created: int = Field(..., description="Number of concept pages created/updated")
    summary: str = Field(..., description="Generated summary of the source")
    message: str = Field(..., description="Status message")


class WikiQueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer from the wiki")
    collection_name: str = Field("main", description="Collection wiki to query")
    file_answer: bool = Field(
        False, description="If True, save the answer as a new wiki page"
    )


class WikiQueryResponse(BaseModel):
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Synthesized answer with citations")
    sources_used: list[str] = Field(default_factory=list, description="Wiki pages cited")
    confidence: str = Field(..., description="Confidence level: high, medium, or low")
    gaps: list[str] = Field(default_factory=list, description="Information gaps")
    suggested_followups: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions or sources"
    )
    filed_page: str | None = Field(None, description="Wiki page path if answer was filed")


class WikiLintRequest(BaseModel):
    collection_name: str = Field("main", description="Collection wiki to lint")
    auto_fix: bool = Field(
        False, description="If True, create stub pages for broken links"
    )


class WikiLintResponse(BaseModel):
    orphan_pages: list[str] = Field(default_factory=list, description="Pages with no inbound links")
    broken_links: list[str] = Field(
        default_factory=list, description="Links pointing to non-existent pages"
    )
    contradictions: list[dict] = Field(default_factory=list, description="Detected contradictions")
    stale_claims: list[dict] = Field(default_factory=list, description="Potentially outdated claims")
    missing_pages: list[dict] = Field(default_factory=list, description="Important terms lacking pages")
    suggestions: list[str] = Field(default_factory=list, description="General improvement suggestions")
    fixes_applied: int = Field(0, description="Number of auto-fixes applied")


class DeleteCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="Collection to delete")


class DeleteDocumentRequest(BaseModel):
    doc_name: str = Field(..., description="Document name to delete")
    collection_name: str = Field("", description="Optional collection filter")


class ReassignDocumentRequest(BaseModel):
    doc_name: str = Field(..., description="Document name to move")
    old_collection: str = Field("", description="Source collection (empty for any)")
    new_collection: str = Field(..., description="Target collection")


class ReassignCollectionRequest(BaseModel):
    old_collection: str = Field(..., description="Current collection name")
    new_collection: str = Field(..., description="New collection name")


class ManageResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    count: int = Field(0, description="Number of affected rows")
    message: str = Field(..., description="Status message")


class EmbedJobAccepted(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    doc_name: str = Field(..., description="Document name being processed")
    collection_name: str = Field(..., description="Target collection")
    status: str = "processing"
    message: str = "Document accepted for async embedding"


class EmbedJobStatus(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="processing, completed, or failed")
    doc_name: str = Field("", description="Document name")
    collection_name: str = Field("", description="Target collection")
    chunks_embedded: int = Field(0, description="Number of chunks embedded")
    error: str | None = Field(None, description="Error message if failed")
    created_at: float | None = Field(None, description="Unix timestamp when job was created")
    finished_at: float | None = Field(None, description="Unix timestamp when job finished")


class EmbedJobListResponse(BaseModel):
    jobs: list[EmbedJobStatus] = Field(default_factory=list)
    processing: int = Field(0, description="Number of active jobs")
    completed: int = Field(0, description="Number of completed jobs")
    failed: int = Field(0, description="Number of failed jobs")


class EmbeddingModelInfo(BaseModel):
    name: str = Field(..., description="Display name of the embedding model")
    code: str = Field(..., description="Model identifier used for embedding")
    is_default: bool = Field(False, description="Whether this is the default model")


class EmbeddingModelsResponse(BaseModel):
    models: list[EmbeddingModelInfo] = Field(..., description="Available embedding models")
    default: str = Field(..., description="Default embedding model code")


class RollbackResponse(BaseModel):
    success: bool = Field(..., description="Whether the rollback succeeded")
    message: str = Field(..., description="Status message")
    backend: str = Field(..., description="Database backend")
    backup_table: str | None = Field(None, description="Name of the backup table used")
    current_model: str = Field("", description="Current embedding model before rollback")
    backup_model: str = Field("", description="Embedding model stored in backup")
    backup_count: int = Field(0, description="Number of records in backup table")
    current_count: int = Field(0, description="Number of records in current table")
    restored_count: int = Field(0, description="Number of records restored")
    dry_run: bool = Field(False, description="Whether this was a dry run")


class BackupTableInfo(BaseModel):
    table_name: str = Field(..., description="Name of the backup table")
    row_count: int = Field(..., description="Number of records in the table")
    embedding_model: str = Field("", description="Embedding model used in the table")
    size_bytes: int | None = Field(None, description="Estimated table size in bytes")
    created_at: str | None = Field(None, description="Approximate creation time")


class BackupTablesResponse(BaseModel):
    backups: list[BackupTableInfo] = Field(..., description="Available backup tables")
    current_table: str = Field(..., description="Name of the active embedding table")
    current_count: int = Field(0, description="Number of records in the active table")
    current_model: str = Field("", description="Active embedding model")
    total_backups: int = Field(0, description="Total number of backup tables found")


class WikiStatusResponse(BaseModel):
    collection: str = Field(..., description="Collection name")
    wiki_path: str = Field(..., description="Filesystem path to the wiki")
    total_pages: int = Field(..., description="Total number of wiki pages")
    page_counts: dict[str, int] = Field(..., description="Breakdown by category")
    orphan_pages: int = Field(..., description="Number of orphan pages")
    broken_links: int = Field(..., description="Number of broken links")
    last_operation: str = Field("", description="Most recent log entry")


class FinalizeRequest(BaseModel):
    collection_name: str = Field(..., description="Original collection name to finalize")
    shadow_collection: str = Field(..., description="Shadow collection name (e.g. 'my_docs_v2')")
    shadow_table: str | None = Field(None, description="Dimension-specific table if rechunk used one")


class FinalizeResponse(BaseModel):
    success: bool = Field(..., description="Whether the finalize succeeded")
    collection: str = Field(..., description="Original collection name")
    original_count: int = Field(0, description="Number of rows in original collection")
    shadow_count: int = Field(0, description="Number of rows in shadow collection")
    deleted_original: int = Field(0, description="Rows deleted from original collection")
    renamed: int = Field(0, description="Rows renamed from shadow to original")
    table_swapped: bool = Field(False, description="Whether a dimension table swap occurred")
    errors: list[str] = Field(default_factory=list, description="Errors encountered")
    message: str = Field("", description="Status message")


class SchemaMigrationResponse(BaseModel):
    backend: str = Field(..., description="Database backend type")
    changes: list[str] = Field(default_factory=list, description="Description of changes made")


class VerifyRequest(BaseModel):
    collection_name: str = Field("", description="Collection to verify (empty for all)")
    dry_run: bool = Field(True, description="Preview only, do not fix")
    auto_finalize: bool = Field(True, description="Finalize any orphaned shadow collections found")
    checks: list[str] | None = Field(
        None,
        description="Specific checks to run (default: all). Options: duplicate_hashes, hash_integrity, "
                    "missing_models, mixed_models, dimension_consistency, partial_documents, "
                    "orphaned_shadows, orphaned_tables, stale_backups, empty_embeddings, "
                    "duplicate_doc_pages, duplicate_content",
    )


class VerifyCheckResult(BaseModel):
    name: str = Field(..., description="Check name")
    severity: str = Field(..., description="error, warning, or info")
    count: int = Field(0, description="Number of issues found")
    details: list[str] = Field(default_factory=list, description="First N examples")
    fix_applied: int | None = Field(None, description="Number of fixes applied (None if dry_run)")


class VerifyResponse(BaseModel):
    collection: str = Field(..., description="Collection verified")
    dry_run: bool = Field(True, description="Whether this was a dry run")
    table: str = Field(..., description="Table checked")
    checks: list[VerifyCheckResult] = Field(default_factory=list, description="Check results")
    summary: dict[str, int] = Field(default_factory=dict, description="Summary counts")
    errors: list[str] = Field(default_factory=list, description="Errors during verification")
    finalized: list[dict] = Field(default_factory=list, description="Results of auto_finalize operations")
