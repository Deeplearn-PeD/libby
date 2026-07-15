from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from libbydbot.api.schemas import (
    WikiBrowseResponse,
    WikiConsolidateRequest,
    WikiConsolidateResponse,
    WikiIngestFromEmbeddingsRequest,
    WikiIngestFromEmbeddingsResponse,
    WikiIngestRequest,
    WikiIngestResponse,
    WikiLintRequest,
    WikiLintResponse,
    WikiPageResponse,
    WikiQueryRequest,
    WikiQueryResponse,
    WikiStatusResponse,
)
from libbydbot.brain.wiki import WikiManager

router = APIRouter(prefix="/wiki", tags=["wiki"])

# Known wiki categories (directories under the wiki root) plus the "root"
# pseudo-category used for index.md / log.md.
WIKI_CATEGORIES = ("sources", "entities", "concepts", "synthesis")


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


@router.post(
    "/ingest-from-embeddings", response_model=WikiIngestFromEmbeddingsResponse
)
def wiki_ingest_from_embeddings(request: WikiIngestFromEmbeddingsRequest):
    """
    Build/update the wiki directly from the embedding table.

    Reconstructs each document's text from its embedded chunks (no PDF
    re-parsing) and ingests it. Useful when the original source files are no
    longer on disk, and as the manual trigger for the same path used by
    automatic post-embedding ingest.
    """
    try:
        from libbydbot.api.main import app_state

        embedder = app_state.embedder
        if embedder is None:
            raise HTTPException(
                status_code=503,
                detail="Embedder not initialized; cannot read embedding table.",
            )

        wiki = get_wiki_manager(request.collection_name)
        result = wiki.ingest_from_embeddings(
            embedder,
            collection=request.collection_name,
            doc_name=request.doc_name,
        )

        per_doc = [
            WikiIngestResponse(
                success=True,
                source=r["source"],
                pages_touched=r["pages_touched"],
                entities_created=r["entities_created"],
                concepts_created=r["concepts_created"],
                summary=r["summary"],
                message=f"Ingested '{r['source']}' from embeddings",
            )
            for r in result.get("results", [])
        ]

        return WikiIngestFromEmbeddingsResponse(
            success=True,
            collection=result["collection"],
            documents_ingested=result["documents_ingested"],
            pages_touched=result["pages_touched"],
            results=per_doc,
            message=(
                f"Ingested {result['documents_ingested']} document(s) "
                f"({result['pages_touched']} pages touched) from embeddings"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting wiki from embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate", response_model=WikiConsolidateResponse)
def wiki_consolidate(request: WikiConsolidateRequest):
    """
    Merge per-part source pages into a single collective page per document.

    Wikis created before per-part merging was supported may contain one
    source page per document part (e.g. ``report_part1``, ``report_part2``).
    This endpoint merges each such group into one page named after the
    original document (``report``) and rewrites inbound wikilinks.
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        result = wiki.consolidate_part_pages()
        return WikiConsolidateResponse(
            success=True,
            collection=request.collection_name,
            groups_merged=result["groups_merged"],
            pages_removed=result["pages_removed"],
            links_rewritten=result["links_rewritten"],
            message=(
                f"Merged {result['groups_merged']} document group(s); "
                f"removed {result['pages_removed']} part page(s); "
                f"rewrote {result['links_rewritten']} link(s)."
            ),
        )
    except Exception as e:
        logger.error(f"Error consolidating wiki: {e}")
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


def _category_dir(wiki: WikiManager, category: str) -> Path:
    """Resolve a category name to its directory on disk."""
    dirs = {
        "sources": wiki.sources_dir,
        "entities": wiki.entities_dir,
        "concepts": wiki.concepts_dir,
        "synthesis": wiki.synthesis_dir,
    }
    if category not in dirs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be one of: {', '.join(dirs)} or 'root'.",
        )
    return dirs[category]


def _safe_resolve(base: Path, *parts: str) -> Path:
    """Resolve *parts* under *base*, rejecting any path that escapes base."""
    candidate = (base.joinpath(*parts)).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Requested path is outside the wiki directory.",
        )
    return candidate


@router.get("/pages/{collection_name}", response_model=WikiBrowseResponse)
def wiki_browse(collection_name: str = "main"):
    """
    Browse a collection's wiki.

    Returns the page names grouped by category (sources/entities/concepts/
    synthesis) plus the root-level pages (index, log), mirroring the TUI tree.
    """
    try:
        wiki = get_wiki_manager(collection_name)
        categories: dict[str, list[str]] = {}
        for category in WIKI_CATEGORIES:
            directory = _category_dir(wiki, category)
            categories[category] = sorted(
                p.stem for p in directory.glob("*.md") if p.is_file()
            )

        root_pages = []
        if wiki.index_path.exists():
            root_pages.append(wiki.index_path.stem)
        if wiki.log_path.exists():
            root_pages.append(wiki.log_path.stem)

        return WikiBrowseResponse(
            collection=wiki.collection_name,
            wiki_path=str(wiki.wiki_dir),
            categories=categories,
            root_pages=root_pages,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error browsing wiki: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/page/{collection_name}", response_model=WikiPageResponse)
def wiki_page(
    collection_name: str = "main",
    category: str = Query(..., description="Category: sources/entities/concepts/synthesis/root"),
    page: str = Query(..., description="Page name (stem without .md), e.g. 'index' or a doc name"),
):
    """
    Read a single wiki page's markdown content.

    Use category='root' with page='index' or page='log' for the root pages.
    """
    try:
        wiki = get_wiki_manager(collection_name)

        if category == "root":
            base = wiki.wiki_dir
        else:
            base = _category_dir(wiki, category)

        # Reject anything that looks like a path component to avoid traversal.
        if not page or "/" in page or "\\" in page or page.startswith("."):
            raise HTTPException(
                status_code=400, detail="page must be a simple page name with no path separators."
            )

        page_path = _safe_resolve(base, f"{page}.md")
        if not page_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Page '{page}' not found in category '{category}'.",
            )

        rel_path = page_path.relative_to(wiki.wiki_dir.resolve())
        content = page_path.read_text(encoding="utf-8")

        return WikiPageResponse(
            collection=wiki.collection_name,
            category=category,
            page=page_path.stem,
            path=str(rel_path),
            content=content,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading wiki page: {e}")
        raise HTTPException(status_code=500, detail=str(e))
