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
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional details")
