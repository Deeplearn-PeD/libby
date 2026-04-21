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


class ModelInfoResponse(BaseModel):
    models: dict[str, dict[str, int]] = Field(
        ..., description="Model usage by collection (model -> collection -> count)"
    )
    total_documents: int = Field(..., description="Total number of documents")


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


class WikiStatusResponse(BaseModel):
    collection: str = Field(..., description="Collection name")
    wiki_path: str = Field(..., description="Filesystem path to the wiki")
    total_pages: int = Field(..., description="Total number of wiki pages")
    page_counts: dict[str, int] = Field(..., description="Breakdown by category")
    orphan_pages: int = Field(..., description="Number of orphan pages")
    broken_links: int = Field(..., description="Number of broken links")
    last_operation: str = Field("", description="Most recent log entry")
