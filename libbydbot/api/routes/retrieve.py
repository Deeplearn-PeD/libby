from collections import Counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from libbydbot.api.schemas import (
    CollectionInfo,
    CollectionListResponse,
    DocumentInfo,
    DocumentListResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedDocument,
)
from libbydbot.brain.embed import DocEmbedder


router = APIRouter(prefix="/api", tags=["retrieval"])


def get_embedder() -> DocEmbedder:
    from libbydbot.api.main import app_state

    return app_state.embedder


EmbedderDep = Annotated[DocEmbedder, Depends(get_embedder)]


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_documents(request: RetrieveRequest, embedder: EmbedderDep):
    """
    Retrieve documents using hybrid search (vector + keyword).

    - **query**: Search query text
    - **collection_name**: Collection to search in (empty for all collections)
    - **num_docs**: Number of documents to retrieve (default: 5, max: 100)
    """
    try:
        results = embedder.retrieve_docs_with_metadata(
            query=request.query,
            collection=request.collection_name,
            num_docs=request.num_docs,
        )

        documents = [
            RetrievedDocument(
                doc_name=r["doc_name"],
                page_number=r["page_number"],
                content=r["content"],
                score=r["score"],
            )
            for r in results
        ]

        return RetrieveResponse(
            query=request.query,
            collection_name=request.collection_name or "all",
            documents=documents,
            total=len(documents),
        )
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    embedder: EmbedderDep,
    collection_name: str = "",
):
    """
    List all embedded documents.

    - **collection_name**: Filter by collection (optional, empty for all)
    """
    try:
        docs = embedder.get_embedded_documents()

        if collection_name:
            docs = [(name, col) for name, col in docs if col == collection_name]

        documents = [
            DocumentInfo(doc_name=name, collection_name=col) for name, col in docs
        ]

        return DocumentListResponse(documents=documents, total=len(documents))
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections", response_model=CollectionListResponse)
def list_collections(embedder: EmbedderDep):
    """
    List all collections with document counts.
    """
    try:
        docs = embedder.get_embedded_documents()

        collection_counts = Counter(col for _, col in docs)

        collections = [
            CollectionInfo(name=name, document_count=count)
            for name, count in collection_counts.items()
        ]

        return CollectionListResponse(collections=collections, total=len(collections))
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))
