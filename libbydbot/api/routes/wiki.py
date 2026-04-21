from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from libbydbot.api.schemas import (
    WikiIngestRequest,
    WikiIngestResponse,
    WikiLintRequest,
    WikiLintResponse,
    WikiQueryRequest,
    WikiQueryResponse,
    WikiStatusResponse,
)
from libbydbot.brain.wiki import WikiManager

router = APIRouter(prefix="/wiki", tags=["wiki"])


def get_wiki_manager(collection_name: str = "main") -> WikiManager:
    """Factory to create a WikiManager for a given collection."""
    from libbydbot.settings import Settings

    try:
        settings = Settings()
    except Exception:
        settings = None

    wiki_base = settings.wiki_base_path if settings else ""
    # Default model from settings if available
    model = "llama3.2"
    if settings and settings.default_model:
        model = settings.default_model

    return WikiManager(
        collection_name=collection_name,
        wiki_base=wiki_base,
        model=model,
    )


@router.post("/ingest", response_model=WikiIngestResponse)
def wiki_ingest(request: WikiIngestRequest):
    """
    Ingest a source document into the wiki.

    The LLM will summarize the source, extract entities and concepts,
    and update the relevant wiki pages.
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        result = wiki.ingest_source(
            doc_name=request.doc_name,
            doc_content=request.doc_content,
            source_type=request.source_type,
        )
        return WikiIngestResponse(
            success=True,
            source=result["source"],
            pages_touched=result["pages_touched"],
            entities_created=result["entities_created"],
            concepts_created=result["concepts_created"],
            summary=result["summary"],
            message=f"Successfully ingested '{request.doc_name}' into wiki '{request.collection_name}'",
        )
    except Exception as e:
        logger.error(f"Error ingesting into wiki: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=WikiQueryResponse)
def wiki_query(request: WikiQueryRequest):
    """
    Query the wiki and synthesize an answer.

    The LLM reads relevant wiki pages and produces a cited answer.
    Optionally files the answer back into the wiki as a new page.
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        result = wiki.query(request.question, file_answer=request.file_answer)
        return WikiQueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources_used=result["sources_used"],
            confidence=result["confidence"],
            gaps=result["gaps"],
            suggested_followups=result["suggested_followups"],
            filed_page=result.get("filed_page"),
        )
    except Exception as e:
        logger.error(f"Error querying wiki: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lint", response_model=WikiLintResponse)
def wiki_lint(request: WikiLintRequest):
    """
    Health-check the wiki.

    Detects orphan pages, broken links, contradictions, stale claims,
    and missing pages. Optionally applies simple auto-fixes.
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        report = wiki.lint(auto_fix=request.auto_fix)
        return WikiLintResponse(
            orphan_pages=report["orphan_pages"],
            broken_links=report["broken_links"],
            contradictions=report["contradictions"],
            stale_claims=report["stale_claims"],
            missing_pages=report["missing_pages"],
            suggestions=report["suggestions"],
            fixes_applied=report.get("fixes_applied", 0),
        )
    except Exception as e:
        logger.error(f"Error linting wiki: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{collection_name}", response_model=WikiStatusResponse)
def wiki_status(collection_name: str = "main"):
    """
    Get statistics about a collection's wiki.
    """
    try:
        wiki = get_wiki_manager(collection_name)
        status = wiki.status()
        return WikiStatusResponse(
            collection=status["collection"],
            wiki_path=status["wiki_path"],
            total_pages=status["total_pages"],
            page_counts=status["page_counts"],
            orphan_pages=status["orphan_pages"],
            broken_links=status["broken_links"],
            last_operation=status["last_operation"],
        )
    except Exception as e:
        logger.error(f"Error getting wiki status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
