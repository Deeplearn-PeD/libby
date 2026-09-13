import threading
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from loguru import logger

from libbydbot.brain import wiki as wiki_brain

from libbydbot.api.schemas import (
    WikiBrowseResponse,
    WikiConsolidateRequest,
    WikiConsolidateResponse,
    WikiExplainRequest,
    WikiExplainResponse,
    WikiGraphQueryRequest,
    WikiGraphQueryResponse,
    WikiGraphRebuildRequest,
    WikiGraphStatusResponse,
    WikiIngestFromEmbeddingsRequest,
    WikiIngestFromEmbeddingsResponse,
    WikiIngestRequest,
    WikiIngestResponse,
    WikiLintRequest,
    WikiLintResponse,
    WikiPageResponse,
    WikiPathRequest,
    WikiPathResponse,
    WikiQueryRequest,
    WikiQueryResponse,
    WikiStatusResponse,
)
from libbydbot.brain.wiki import WikiManager

router = APIRouter(prefix="/wiki", tags=["wiki"])

# Known wiki categories (directories under the wiki root) plus the "root"
# pseudo-category used for index.md / log.md.
WIKI_CATEGORIES = ("sources", "entities", "concepts", "synthesis")

#: Per-collection WikiManager cache. WikiManager.__init__ builds LLM clients
#: and (via the shared graph cache) is the unit of reuse, so graph endpoints
#: don't reconstruct managers — or LLM clients — on every request.
_WIKI_MANAGERS: dict[tuple[str, str], WikiManager] = {}
_WIKI_MANAGERS_LOCK = threading.Lock()


def get_wiki_manager(collection_name: str = "main") -> WikiManager:
    """Return a cached WikiManager for a given collection."""
    from libbydbot.settings import Settings

    try:
        settings = Settings()
    except Exception:
        settings = None

    wiki_base = settings.wiki_base_path if settings else ""
    # Dedicated wiki model if configured, else the default chat model.
    model = "llama3.2"
    if settings:
        model = settings.wiki_model or settings.default_model or model

    key = (str(wiki_base), collection_name)
    with _WIKI_MANAGERS_LOCK:
        manager = _WIKI_MANAGERS.get(key)
        if manager is None:
            manager = WikiManager(
                collection_name=collection_name,
                wiki_base=wiki_base,
                model=model,
            )
            _WIKI_MANAGERS[key] = manager
        return manager


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

        docs_ingested = result.get("documents_ingested", 0)
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
            # a run that finds no embedded documents is reported as a
            # failure (with reason) so callers do not mistake it for a
            # successful no-op
            success=docs_ingested > 0,
            collection=result["collection"],
            documents_ingested=docs_ingested,
            pages_touched=result.get("pages_touched", 0),
            results=per_doc,
            errors=result.get("errors", []),
            reason=result.get("reason"),
            message=(
                f"Ingested {docs_ingested} document(s) "
                f"({result.get('pages_touched', 0)} pages touched) from embeddings"
                if docs_ingested
                else result.get("reason", "No embedded documents found")
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting wiki from embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingest-diagnosis/{collection_name}")
def wiki_ingest_diagnosis(collection_name: str = "main"):
    """
    Diagnose why (or whether) a collection can build wiki pages.

    Reports, per embedding table: how many rows of this collection it
    holds; the document count after cross-table merge; how many wiki
    pages exist on disk; and the knowledge-graph size. Read-only.
    """
    try:
        from libbydbot.api.main import app_state

        embedder = app_state.embedder
        if embedder is None:
            raise HTTPException(
                status_code=503,
                detail="Embedder not initialized; cannot read embedding table.",
            )

        wiki = get_wiki_manager(collection_name)
        kg = wiki._get_knowledge_graph()

        tables = []
        docs_by_name: set[str] = set()
        for tbl in embedder.candidate_text_tables():
            try:
                rows = embedder._fetch_doc_rows(tbl, collection_name, "")
            except Exception as e:
                tables.append({"table": tbl, "error": str(e)})
                continue
            names = {r[0] for r in rows}
            docs_by_name.update(names)
            tables.append(
                {
                    "table": tbl,
                    "rows": len(rows),
                    "documents": len(names),
                }
            )

        pages_on_disk = [
            str(p.relative_to(wiki.wiki_dir))
            for p in wiki.wiki_dir.rglob("*.md")
            if p.name not in ("index.md", "log.md")
        ]

        return {
            "collection": collection_name,
            "embedding_tables": tables,
            "merged_documents": len(docs_by_name),
            "document_names": sorted(docs_by_name),
            "wiki_dir": str(wiki.wiki_dir),
            "wiki_pages_on_disk": len(pages_on_disk),
            "wiki_pages": sorted(pages_on_disk),
            "graph": {
                "nodes": kg.graph.number_of_nodes(),
                "edges": kg.graph.number_of_edges(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error diagnosing wiki ingest for '{collection_name}': {e}")
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
            graph=status.get("graph"),
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


@router.post("/graph/rebuild", response_model=WikiGraphStatusResponse)
def wiki_graph_rebuild(request: WikiGraphRebuildRequest):
    """
    Rebuild the wiki knowledge graph from the pages on disk.

    Chunk nodes and their reference edges are preserved across rebuilds.
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        status = wiki.graph_rebuild()
        return WikiGraphStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error rebuilding wiki graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{collection_name}", response_model=WikiGraphStatusResponse)
def wiki_graph_status(collection_name: str = "main"):
    """
    Get statistics about a collection's knowledge graph.

    ``rebuilding`` is true while a background rebuild is in flight; the
    statistics describe the previous (still-served) snapshot.
    """
    try:
        wiki = get_wiki_manager(collection_name)
        status = wiki.graph_status()
        status["rebuilding"] = wiki_brain.graph_rebuilding(wiki.wiki_dir)
        return WikiGraphStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting wiki graph status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{collection_name}/viz")
def wiki_graph_viz(
    collection_name: str = "main",
    rebuild: bool = False,
    include_chunks: bool = False,
):
    """
    Serve the interactive knowledge graph visualization (graph.html shell).

    The shell is a small document that renders the top hubs immediately;
    the remaining nodes/edges stream in via ``/graph/{c}/data`` batches.
    When the export is stale relative to the wiki, a single-flight
    background rebuild refreshes it while the previous shell keeps being
    served (``X-Graph-Rebuilding: 1`` response header). Pass
    ``?rebuild=true`` to force an async rebuild. Chunk nodes — one per
    embedded document piece, typically the bulk of the graph — are
    excluded from batches by default; pass ``include_chunks=true`` to the
    data endpoint to restore them.
    """
    try:
        wiki = get_wiki_manager(collection_name)
        viz = wiki.graph_viz_html(rebuild=rebuild, include_chunks=include_chunks)
        headers = {}
        if viz.get("rebuilding"):
            headers["X-Graph-Rebuilding"] = "1"
        return FileResponse(
            viz["path"],
            media_type="text/html",
            headers=headers or None,
            content_disposition_type="inline",
        )
    except Exception as e:
        logger.error(f"Error serving wiki graph visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{collection_name}/data")
def wiki_graph_data(
    collection_name: str = "main",
    cursor: int = Query(0, ge=0, description="Offset into the degree-ranked node list"),
    limit: int = Query(250, ge=1, le=1000, description="Max nodes per page"),
    include_chunks: bool = Query(False, description="Include chunk nodes in batches"),
):
    """
    Return one page of visualization data for incremental rendering.

    Nodes arrive degree-ranked (hubs first) with precomputed positions so
    the browser places them without a physics simulation. Every edge is
    delivered exactly once, with the batch in which its later endpoint
    arrives. All batches of a walk carry the same ``snapshot_id``; when it
    changes (a background rebuild swapped the graph), clients restart the
    walk from cursor 0.
    """
    try:
        wiki = get_wiki_manager(collection_name)
        page = wiki.graph_data_page(
            cursor=cursor, limit=limit, include_chunks=include_chunks
        )
        page["rebuilding"] = wiki_brain.graph_rebuilding(wiki.wiki_dir)
        return page
    except Exception as e:
        logger.error(f"Error serving wiki graph data page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph-lib/{filename}")
def wiki_graph_lib(filename: str):
    """
    Serve a static visualization library asset (no auth, inert JS).

    These files are large and version-stable, so they are served with
    immutable caching — clients fetch them once instead of re-downloading
    them embedded in every graph export.
    """
    allowed = {
        "vis-network.min.js": ("pyvis", "lib", "vis-9.1.2", "vis-network.min.js"),
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Unknown graph library asset")
    import importlib
    import importlib.resources

    parts = allowed[filename]
    try:
        import importlib

        mod = importlib.import_module(parts[0])
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            raise FileNotFoundError(parts[0])
        lib_path = Path(mod_file).parent.joinpath(*parts[1:])
        if not lib_path.is_file():
            raise FileNotFoundError(lib_path)
    except Exception as e:
        logger.error(f"Graph library asset missing: {e}")
        raise HTTPException(status_code=404, detail="Graph library asset unavailable")
    return FileResponse(
        lib_path,
        media_type="application/javascript",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@router.post("/graph/path", response_model=WikiPathResponse)
def wiki_graph_path(request: WikiPathRequest):
    """
    Find the shortest path between two nodes in the knowledge graph.
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        result = wiki.graph_path(request.source, request.target)
        return WikiPathResponse(
            found=result["found"],
            length=result.get("length", 0),
            nodes=result.get("nodes", []),
            edges=result.get("edges", []),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"Error finding wiki graph path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/explain", response_model=WikiExplainResponse)
def wiki_graph_explain(request: WikiExplainRequest):
    """
    Explain a node in the knowledge graph (attributes + connections).
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        result = wiki.graph_explain(request.name)
        return WikiExplainResponse(
            found=result["found"],
            id=result.get("id", ""),
            title=result.get("title", ""),
            node_type=result.get("node_type", ""),
            degree=result.get("degree", 0),
            outbound=result.get("outbound", []),
            inbound=result.get("inbound", []),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"Error explaining wiki graph node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/query", response_model=WikiGraphQueryResponse)
def wiki_graph_query(request: WikiGraphQueryRequest):
    """
    Return the ranked subgraph relevant to a question.
    """
    try:
        wiki = get_wiki_manager(request.collection_name)
        result = wiki.graph_query(request.question, max_nodes=request.max_nodes)
        return WikiGraphQueryResponse(
            question=result["question"],
            nodes=result["nodes"],
            edges=result["edges"],
        )
    except Exception as e:
        logger.error(f"Error querying wiki graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
