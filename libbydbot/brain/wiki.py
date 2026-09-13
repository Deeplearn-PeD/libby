"""
LLM Wiki manager for Libby D. Bot.

The WikiManager maintains a persistent, compounding markdown knowledge base
for each document collection. It handles ingest (integrating sources into the wiki),
query (synthesizing answers from wiki pages), and lint (health-checking the wiki).

Wikis are stored as Obsidian-compatible markdown files with YAML frontmatter
and [[wikilink]] cross-references.
"""

import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import loguru
import yaml

from libbydbot.brain.graph import is_shell_html
from libbydbot.brain.wiki_models import (
    LintReport,
    SourceSummary,
    WikiQueryAnswer,
    WikiUpdatePlan,
)
from base_agent.llminterface import LangModel, StructuredLangModel

logger = loguru.logger

DEFAULT_WIKI_BASE = Path.home() / ".libby" / "wikis"


# ───────────────────────── knowledge-graph cache ─────────────────────────
#
# One shared entry per wiki directory: the parsed WikiKnowledgeGraph
# survives across requests instead of being re-read from graph.json on
# every API call. The cached instance is treated as read-only — ingest
# work and rebuilds happen on private instances that atomically replace
# the cached one — so API readers never observe in-place mutation or
# half-written state. Rebuilds are single-flight per collection and run
# in a background thread; readers keep using the previous (stale) graph
# until the worker swaps in the fresh one (stale-while-revalidate).


@dataclass
class _GraphCacheEntry:
    """Shared per-collection knowledge-graph state."""

    kg: Any = None
    graph_mtime_ns: int = 0
    rebuilding: bool = False
    dirty: bool = False
    #: queued (doc_name, summary, chunks) ingests applied by the rebuild worker
    pending: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_GRAPH_CACHE: dict[str, _GraphCacheEntry] = {}
_GRAPH_CACHE_GUARD = threading.Lock()


def _graph_entry(wiki_dir: str | Path) -> _GraphCacheEntry:
    key = str(wiki_dir)
    with _GRAPH_CACHE_GUARD:
        entry = _GRAPH_CACHE.get(key)
        if entry is None:
            entry = _GraphCacheEntry()
            _GRAPH_CACHE[key] = entry
        return entry


def _graph_json_mtime(kg) -> int:
    try:
        return kg.graph_path.stat().st_mtime_ns
    except OSError:
        return 0


def get_cached_kg(wiki_dir: str | Path, collection_name: str):
    """Return the cached WikiKnowledgeGraph, reloading when graph.json changed."""
    from libbydbot.brain.graph import WikiKnowledgeGraph

    entry = _graph_entry(wiki_dir)
    with entry.lock:
        kg = entry.kg
        if kg is not None and _graph_json_mtime(kg) == entry.graph_mtime_ns:
            return kg
        kg = WikiKnowledgeGraph(wiki_dir, collection_name)
        entry.kg = kg
        entry.graph_mtime_ns = _graph_json_mtime(kg)
        return kg


def run_graph_rebuild(wiki_dir: str | Path, collection_name: str):
    """Rebuild body shared by the background worker and the admin sync path.

    Builds a fresh private instance (loading the latest graph.json), applies
    queued ingest updates, rescans the pages, refreshes the shell export,
    and swaps the result into the cache.
    """
    from libbydbot.brain.graph import WikiKnowledgeGraph

    entry = _graph_entry(wiki_dir)
    with entry.lock:
        entry.dirty = False
        pending, entry.pending = entry.pending, []

    kg = WikiKnowledgeGraph(wiki_dir, collection_name)
    for doc_name, summary, chunks in pending:
        try:
            kg.update_from_ingest(doc_name, summary, chunks)
        except Exception as e:
            logger.warning(f"Queued graph ingest failed for '{doc_name}': {e}")
    kg.rebuild()
    kg.export_shell()
    with entry.lock:
        entry.kg = kg
        entry.graph_mtime_ns = _graph_json_mtime(kg)
    return kg


def _rebuild_worker(wiki_dir: str, collection_name: str) -> None:
    entry = _graph_entry(wiki_dir)
    try:
        while True:
            kg = run_graph_rebuild(wiki_dir, collection_name)
            logger.info(
                f"Background graph rebuild done for '{collection_name}' "
                f"({kg.graph.number_of_nodes()} nodes, "
                f"{kg.graph.number_of_edges()} edges)"
            )
            with entry.lock:
                again = entry.dirty
                if not again:
                    entry.rebuilding = False
                    return
    except Exception as e:
        logger.error(f"Background graph rebuild failed for '{collection_name}': {e}")
        with entry.lock:
            entry.rebuilding = False


def schedule_graph_rebuild(wiki_dir: str | Path, collection_name: str) -> bool:
    """Start a background rebuild unless one is already running (single-flight).

    Returns True when a new worker was started; when one is already running
    the request is recorded (``dirty``) and the worker repeats after finishing.
    """
    entry = _graph_entry(wiki_dir)
    with entry.lock:
        if entry.rebuilding:
            entry.dirty = True
            return False
        entry.rebuilding = True
    threading.Thread(
        target=_rebuild_worker,
        args=(str(wiki_dir), collection_name),
        name=f"kg-rebuild-{collection_name}",
        daemon=True,
    ).start()
    return True


def graph_rebuilding(wiki_dir: str | Path) -> bool:
    """True while a background rebuild for this wiki is in flight."""
    entry = _graph_entry(wiki_dir)
    with entry.lock:
        return entry.rebuilding

INDEX_TEMPLATE = """# Wiki Index

> Catalog of all pages in this wiki. Updated automatically on ingest.

## Sources

{sources}

## Entities

{entities}

## Concepts

{concepts}

## Synthesis

{synthesis}

---
*Last updated: {timestamp}*
"""

LOG_TEMPLATE = """# Wiki Log

> Chronological record of all wiki operations.
> Parseable with: `grep "^## \\[" log.md`

"""

SOURCE_PAGE_TEMPLATE = """---
title: {title}
date_ingested: {date}
source_type: {source_type}
---

# {title}

## Summary

{summary}

## Key Takeaways

{takeaways}

## Entities Mentioned

{entities}

## Concepts Discussed

{concepts}

## Questions Raised

{questions}

## Raw Source

See collection: `{collection}`
"""

ENTITY_PAGE_TEMPLATE = """---
title: {title}
entity_type: {entity_type}
date_created: {date}
---

# {title}

{description}

## Mentions in Sources

{mentions}

## Related Entities

{related}
"""

CONCEPT_PAGE_TEMPLATE = """---
title: {title}
date_created: {date}
---

# {title}

{description}

## Mentions in Sources

{mentions}

## Related Concepts

{related}
"""


class WikiManager:
    """
    Manages a markdown wiki for a single document collection.

    The wiki lives on the filesystem as a directory of markdown files
    with Obsidian-compatible wikilinks and YAML frontmatter.
    """

    # Patterns that identify a document "part" suffix, e.g. "report_part1",
    # "report part 2", "report_p3", "report (1)", "report_vol2". The first
    # capture group is the part index used for ordering when merging.
    _PART_PATTERNS: list[re.Pattern] = [
        re.compile(
            r"[\s_-]+(?:part|pt|p|vol(?:ume)?|chapter|ch|section|sec)[\s_-]*(\d+)\s*$",
            re.IGNORECASE,
        ),
        re.compile(r"\s*\(\s*(?:part\s+)?(\d+)\s*\)\s*$", re.IGNORECASE),
    ]

    def __init__(
        self,
        collection_name: str,
        wiki_base: str | Path = "",
        model: str = "kimi-k2.5",
        graph_enabled: bool | None = None,
    ):
        self.collection_name = collection_name
        self.wiki_base = Path(wiki_base) if wiki_base else DEFAULT_WIKI_BASE
        self.wiki_dir = self.wiki_base / self._sanitize_name(collection_name)
        self.model = model
        self._struct_llm = StructuredLangModel(model=model)
        self._llm = LangModel(model=model)
        self._structured_output_supported: bool | None = None
        if graph_enabled is None:
            try:
                from libbydbot.settings import Settings

                graph_enabled = Settings().wiki_graph_enabled
            except Exception:
                graph_enabled = True
        self.graph_enabled = graph_enabled
        self._ensure_structure()

    # ────────────────────────── knowledge graph ─────────────────────

    def _get_knowledge_graph(self):
        """Return the shared cached WikiKnowledgeGraph for this wiki."""
        return get_cached_kg(self.wiki_dir, self.collection_name)

    def _fetch_document_chunks(self, doc_name: str, doc_content: str) -> list[dict]:
        """
        Fetch embedded chunks for a document from the embedding database.

        Also gathers chunks stored under per-part names when the document was
        merged from parts at ingest time (e.g. ``report`` built from
        ``report_part1``/``report_part2``), so chunk references keep pointing
        at the real embedded chunks. Falls back to chunking the raw content
        when the document is not embedded at all.
        """
        try:
            from libbydbot.brain.embed import DocEmbedder
            from libbydbot.settings import Settings

            # EMBED_DB is used by the API server and docker deployments;
            # fall back to the settings embed_db_url otherwise.
            dburl = os.getenv("EMBED_DB", "") or Settings().embed_db_url
            embedder = DocEmbedder(col_name=self.collection_name, dburl=dburl)
            chunks = embedder.get_document_chunks(doc_name)
            if chunks:
                return chunks

            # Document may have been merged from parts; collect the chunks of
            # every part that reduces to this base name.
            part_chunks = self._fetch_part_chunks(embedder, doc_name)
            if part_chunks:
                return part_chunks
        except Exception as e:
            logger.warning(f"Could not fetch embedded chunks for '{doc_name}': {e}")

        from hashlib import sha256

        from libbydbot.brain.ingest import TextSplitter

        splitter = TextSplitter()
        return [
            {
                "doc_hash": sha256(chunk.encode()).hexdigest(),
                "doc_name": doc_name,
                "page_number": i,
                "content": chunk,
            }
            for i, chunk in enumerate(splitter.split_text(doc_content))
        ]

    def _fetch_part_chunks(self, embedder, doc_name: str) -> list[dict]:
        """Collect chunks of all embedded parts that merge into *doc_name*."""
        try:
            embedded_names = {
                name
                for name, collection in embedder.get_embedded_documents()
                if not collection or collection == self.collection_name
            }
        except Exception as e:
            logger.warning(f"Could not list embedded documents: {e}")
            return []

        part_names = [
            name
            for name in embedded_names
            if name != doc_name and self._doc_base_and_part(name)[0] == doc_name
        ]
        if not part_names:
            return []

        def part_sort_key(name: str):
            part = self._doc_base_and_part(name)[1]
            return (part if part is not None else float("inf"), name)

        chunks: list[dict] = []
        for name in sorted(part_names, key=part_sort_key):
            chunks.extend(embedder.get_document_chunks(name))
        return chunks

    def _update_knowledge_graph(self, doc_name: str, summary, doc_content: str) -> None:
        """Queue a knowledge-graph refresh after an ingest (applied async).

        Chunks are fetched immediately (the embedding DB state matches the
        ingest) while the incremental update and full page rescan run in the
        background rebuild worker; readers keep using the previous graph
        until the worker swaps in the fresh one.
        """
        if not self.graph_enabled:
            return
        try:
            chunks = self._fetch_document_chunks(doc_name, doc_content)
            entry = _graph_entry(self.wiki_dir)
            with entry.lock:
                entry.pending.append((doc_name, summary, chunks))
            schedule_graph_rebuild(self.wiki_dir, self.collection_name)
        except Exception as e:
            logger.warning(f"Knowledge graph update failed for '{doc_name}': {e}")

    def graph_rebuild(self) -> dict:
        """Rebuild the knowledge graph from the wiki pages on disk (synchronous)."""
        kg = run_graph_rebuild(self.wiki_dir, self.collection_name)
        self._append_log("graph", "rebuilt knowledge graph", kg.graph.number_of_nodes())
        return kg.status()

    def graph_status(self) -> dict:
        """Return knowledge graph statistics."""
        return self._get_knowledge_graph().status()

    def graph_path(self, a: str, b: str) -> dict:
        """Find the shortest path between two nodes in the knowledge graph."""
        return self._get_knowledge_graph().shortest_path(a, b)

    def graph_explain(self, name: str) -> dict:
        """Explain a node in the knowledge graph (attributes + connections)."""
        return self._get_knowledge_graph().explain(name)

    def graph_query(self, question: str, max_nodes: int = 15) -> dict:
        """Return the ranked subgraph relevant to a question."""
        return self._get_knowledge_graph().subgraph_for_query(question, max_nodes=max_nodes)

    def graph_data_page(
        self, cursor: int, limit: int, include_chunks: bool = False
    ) -> dict:
        """Return one page of incremental viz data from the cached snapshot."""
        return self._get_knowledge_graph().viz_data_page(
            cursor, limit, include_chunks=include_chunks
        )

    def graph_neighbors_html(
        self, name: str, depth: int = 1, include_chunks: bool = False
    ) -> dict:
        """Render the ego-centered neighborhood page for a node."""
        kg = self._get_knowledge_graph()
        if kg.graph.number_of_nodes() == 0:
            kg.rebuild()
        return kg.neighbors_html(name, depth=depth, include_chunks=include_chunks)

    def flush_graph_updates(self) -> dict:
        """Apply queued ingest updates and rebuild synchronously.

        The background worker normally does this; tests and admin tools use
        this to force the graph up to date without waiting on the thread.
        """
        kg = run_graph_rebuild(self.wiki_dir, self.collection_name)
        return kg.status()

    def graph_export_html(self, path: str | Path | None = None) -> Path:
        """Export an interactive HTML visualization of the knowledge graph."""
        return self._get_knowledge_graph().export_html(path)

    def graph_viz_html(
        self,
        rebuild: bool = False,
        include_chunks: bool = False,
    ) -> dict:
        """Return ``{"path", "rebuilding"}`` for the interactive visualization.

        The shell page (``graph.html``) is reused while fresh; when stale —
        or when ``rebuild=True`` forces a refresh — a single-flight
        background rebuild is scheduled and the previous shell keeps being
        served until the worker swaps in the new one (stale-while-
        revalidate). A missing shell is built synchronously once so the
        endpoint always returns something renderable. ``include_chunks``
        is honored by the ``/graph/{c}/data`` batch endpoint, not the shell.
        """
        kg = self._get_knowledge_graph()
        html_path = kg.wiki_dir / "graph.html"

        if not html_path.exists() or not is_shell_html(html_path):
            # missing, or a legacy pyvis export from before the shell viz:
            # export the shell synchronously (fast — no full rebuild needed)
            if kg.graph.number_of_nodes() == 0:
                kg.rebuild()
            kg.export_shell()
        elif rebuild or self._viz_cache_stale(kg, html_path):
            if rebuild:
                entry = _graph_entry(self.wiki_dir)
                with entry.lock:
                    entry.dirty = True
            scheduled = schedule_graph_rebuild(self.wiki_dir, self.collection_name)
            if scheduled or rebuild:
                logger.debug(
                    "Knowledge-graph visualization stale — background "
                    "rebuild scheduled, serving previous shell"
                )
        else:
            logger.debug("Serving cached knowledge-graph visualization")
        return {
            "path": html_path,
            "rebuilding": graph_rebuilding(self.wiki_dir),
        }

    @staticmethod
    def _viz_cache_stale(kg, html_path: Path) -> bool:
        """True when graph.html is missing or older than the wiki state."""
        if not html_path.exists():
            return True
        try:
            html_mtime = html_path.stat().st_mtime
            if kg.graph_path.exists() and kg.graph_path.stat().st_mtime > html_mtime:
                return True
            for md_file in kg.wiki_dir.rglob("*.md"):
                if md_file.stat().st_mtime > html_mtime:
                    return True
        except OSError as e:
            logger.warning(f"Could not stat graph cache files: {e}")
            return True
        return False

    def _graph_rank_pages(self, question: str) -> list[str]:
        """Rank wiki pages for a question using the knowledge graph."""
        if not self.graph_enabled:
            return []
        try:
            kg = self._get_knowledge_graph()
            if kg.graph.number_of_nodes() == 0:
                return []
            return kg.score_pages(question)
        except Exception as e:
            logger.warning(f"Graph-based page ranking failed: {e}")
            return []

    # ────────────────────────── properties ──────────────────────────

    @property
    def index_path(self) -> Path:
        return self.wiki_dir / "index.md"

    @property
    def log_path(self) -> Path:
        return self.wiki_dir / "log.md"

    @property
    def sources_dir(self) -> Path:
        return self.wiki_dir / "sources"

    @property
    def entities_dir(self) -> Path:
        return self.wiki_dir / "entities"

    @property
    def concepts_dir(self) -> Path:
        return self.wiki_dir / "concepts"

    @property
    def synthesis_dir(self) -> Path:
        return self.wiki_dir / "synthesis"

    # ────────────────────────── setup ───────────────────────────────

    def _sanitize_name(self, name: str) -> str:
        """Convert a collection name to a filesystem-safe directory name."""
        return re.sub(r"[^\w\-]", "_", name).lower()

    def _ensure_structure(self) -> None:
        """Create the wiki directory structure if it doesn't exist."""
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        for subdir in (
            self.sources_dir,
            self.entities_dir,
            self.concepts_dir,
            self.synthesis_dir,
        ):
            subdir.mkdir(exist_ok=True)

        if not self.index_path.exists():
            self._write_index()

        if not self.log_path.exists():
            self.log_path.write_text(LOG_TEMPLATE, encoding="utf-8")

        logger.info(f"Wiki ready at {self.wiki_dir}")

    # ────────────────────────── file I/O ────────────────────────────

    def _read_page(self, path: Path) -> str:
        """Read a markdown page, returning empty string if missing."""
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_page(self, path: Path, content: str) -> None:
        """Write content to a markdown page, creating parent dirs if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote page: {path}")

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Extract YAML frontmatter and body from markdown content."""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    fm = {}
                return fm, parts[2].strip()
        return {}, content.strip()

    def _build_frontmatter(self, data: dict[str, Any]) -> str:
        """Build a YAML frontmatter block from a dict."""
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False)
        return f"---\n{yaml_str}---\n\n"

    # ────────────────────────── index / log ─────────────────────────

    def _write_index(self) -> None:
        """Rebuild index.md from the current state of the wiki."""
        timestamp = datetime.now().isoformat()
        sources = self._list_pages(self.sources_dir, "source")
        entities = self._list_pages(self.entities_dir, "entity")
        concepts = self._list_pages(self.concepts_dir, "concept")
        synthesis = self._list_pages(self.synthesis_dir, "synthesis")

        content = INDEX_TEMPLATE.format(
            sources=sources,
            entities=entities,
            concepts=concepts,
            synthesis=synthesis,
            timestamp=timestamp,
        )
        self._write_page(self.index_path, content)

    def _list_pages(self, directory: Path, page_type: str) -> str:
        """Generate a markdown list of pages in a directory with one-line summaries."""
        lines = []
        for path in sorted(directory.glob("*.md")):
            title = path.stem
            content = self._read_page(path)
            _, body = self._parse_frontmatter(content)
            # Try to extract first sentence as summary
            first_line = body.split("\n")[0] if body else ""
            if first_line.startswith("# "):
                # Skip blank lines after the header to find actual content
                lines = body.split("\n")
                first_line = ""
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped:
                        first_line = stripped
                        break
            summary = first_line.strip("- ")[:120]
            lines.append(f"- [[{title}]] — {summary}")
        if not lines:
            lines.append(f"_No {page_type} pages yet._")
        return "\n".join(lines)

    def _append_log(self, operation: str, detail: str, pages_touched: int = 0) -> None:
        """Append an entry to log.md."""
        entry = (
            f"## [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {operation} | {detail}"
        )
        if pages_touched:
            entry += f" | touched {pages_touched} pages"
        entry += "\n\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    # ────────────────────────── link graph ──────────────────────────

    def _extract_wikilinks(self, content: str) -> list[str]:
        """Extract all [[wikilink]] targets from markdown content."""
        return re.findall(r"\[\[([^\]]+)\]\]", content)

    def _build_link_graph(self) -> dict[str, list[str]]:
        """Build a mapping of page path -> list of outbound wikilinks."""
        graph: dict[str, list[str]] = {}
        for md_file in self.wiki_dir.rglob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            content = self._read_page(md_file)
            links = self._extract_wikilinks(content)
            rel_path = md_file.relative_to(self.wiki_dir).as_posix()
            graph[rel_path] = links
        return graph

    def _find_orphans(self) -> list[str]:
        """Find pages with no inbound wikilinks."""
        graph = self._build_link_graph()
        all_pages = set(graph.keys())
        linked_pages: set[str] = set()
        for links in graph.values():
            for link in links:
                # Try to resolve link to a relative path (case-insensitive)
                link_lower = link.lower()
                for page in all_pages:
                    if Path(page).stem.lower() == link_lower or page.lower() == link_lower:
                        linked_pages.add(page)
                        break
        return sorted(all_pages - linked_pages)

    def _find_broken_links(self) -> list[str]:
        """Find wikilinks that point to non-existent pages."""
        graph = self._build_link_graph()
        all_stems = {Path(p).stem.lower() for p in graph.keys()}
        broken: set[str] = set()
        for links in graph.values():
            for link in links:
                if link.lower() not in all_stems:
                    broken.add(link)
        return sorted(broken)

    # ────────────────────────── part detection ──────────────────────

    @classmethod
    def _doc_base_and_part(cls, doc_name: str) -> tuple[str, int | None]:
        """
        Strip a document "part" suffix and return the base name + part index.

        Recognizes suffixes such as ``_part1``, `` part 2``, ``_p3``,
        ``(4)``, ``_vol2``, ``_chapter1``. Returns ``(doc_name, None)`` when
        no part suffix is detected so that standalone documents are left
        untouched.
        """
        for pat in cls._PART_PATTERNS:
            m = pat.search(doc_name)
            if m:
                base = doc_name[: m.start()].rstrip(" _-")
                if not base:
                    # The entire name was the suffix; keep the original.
                    return doc_name, None
                try:
                    part = int(m.group(1))
                except (ValueError, IndexError):
                    part = None
                return base, part
        return doc_name, None

    @staticmethod
    def _group_documents_by_base(texts: dict[str, str]) -> dict[str, str]:
        """
        Merge documents that are parts of the same source into one entry.

        Takes a ``{doc_name: text}`` mapping (as returned by
        ``get_document_texts``) and returns a mapping keyed by the *base*
        document name, with the text of each group's parts concatenated in
        ascending part-index order. Singleton documents are passed through
        unchanged, except that a lone part (e.g. ``report_part1`` with no
        siblings) is renamed to its base name so the wiki page reflects the
        original document name.
        """
        buckets: dict[str, list[tuple[int | None, str, str]]] = {}
        order: list[str] = []
        for name, content in texts.items():
            base, part = WikiManager._doc_base_and_part(name)
            if base not in buckets:
                buckets[base] = []
                order.append(base)
            buckets[base].append((part, name, content))

        merged: dict[str, str] = {}
        for base in order:
            items = buckets[base]
            if len(items) == 1:
                part, name, content = items[0]
                # A lone part still gets renamed to its base document name.
                key = base if part is not None else name
                merged[key] = content
            else:
                items.sort(
                    key=lambda t: (t[0] if t[0] is not None else float("inf"), t[1])
                )
                merged[base] = "\n\n".join(content for _, _, content in items)
        return merged

    # ────────────────────────── ingest ──────────────────────────────

    def ingest_source(
        self,
        doc_name: str,
        doc_content: str,
        source_type: str = "document",
    ) -> dict[str, Any]:
        """
        Integrate a source document into the wiki.

        Steps:
        1. Generate structured summary of the source.
        2. Plan which wiki pages to create/update.
        3. Write/update pages.
        4. Update index and log.
        """
        logger.info(f"Ingesting source into wiki: {doc_name}")

        # 1. Structured summary
        summary = self._generate_source_summary(doc_name, doc_content)

        # 2. Update plan
        plan = self._generate_update_plan(doc_name, summary)

        # 3. Apply updates
        pages_touched = 0

        # Write source page
        source_page = self._build_source_page(doc_name, summary, source_type)
        source_path = self.sources_dir / f"{self._sanitize_name(doc_name)}.md"
        self._write_page(source_path, source_page)
        pages_touched += 1

        # Write/update entity pages
        for entity in summary.entities:
            pages_touched += self._update_entity_page(entity, doc_name)

        # Write/update concept pages
        for concept in summary.concepts:
            pages_touched += self._update_concept_page(concept, doc_name)

        # Update synthesis if indicated
        if plan.synthesis_notes:
            pages_touched += self._update_synthesis(plan.synthesis_notes, doc_name)

        # 4. Rebuild index and log
        self._write_index()
        self._append_log("ingest", doc_name, pages_touched)

        # 5. Refresh the knowledge graph
        self._update_knowledge_graph(doc_name, summary, doc_content)

        logger.success(f"Wiki ingest complete: {doc_name} ({pages_touched} pages touched)")
        return {
            "source": doc_name,
            "pages_touched": pages_touched,
            "entities_created": len(summary.entities),
            "concepts_created": len(summary.concepts),
            "summary": summary.summary,
        }

    def ingest_from_embeddings(
        self,
        embedder,
        collection: str = "",
        doc_name: str = "",
        merge_parts: bool = True,
    ) -> dict[str, Any]:
        """
        Build/update the wiki directly from the embedding table.

        Reconstructs each document's text from its embedded chunks (ordered by
        page number) and feeds it through :meth:`ingest_source`. This avoids
        re-parsing the original PDFs and works even when the source files are
        no longer on disk.

        When *merge_parts* is ``True`` (the default), documents that were split
        into parts at embedding time — and therefore stored under several
        ``doc_name`` values that share a part suffix (e.g. ``report_part1``,
        ``report_part2``) — are concatenated back into a single source and
        produce **one** wiki page named after the original document
        (``report``).

        :param embedder: a ``DocEmbedder`` instance backed by the same store
            as the embedded collection.
        :param collection: collection to read; defaults to this wiki's collection.
        :param doc_name: ingest only this document; empty means all documents.
        :param merge_parts: merge per-part documents into one page each.
        """
        collection = collection or self.collection_name
        texts = embedder.get_document_texts(collection=collection, doc_name=doc_name)
        if not texts:
            reason = (
                f"No embedded documents found for collection '{collection}' — "
                "either the collection has no embeddings, or the embedding "
                "table holding it is not visible to this Libby build "
                "(upgrade if the store was migrated across backends)."
            )
            logger.warning(
                f"ingest_from_embeddings: {reason} "
                f"(tables checked: {embedder.candidate_text_tables()})"
            )
            return {
                "success": False,
                "reason": reason,
                "tables_checked": embedder.candidate_text_tables(),
                "collection": collection,
                "documents_ingested": 0,
                "pages_touched": 0,
                "results": [],
            }

        # Guard against empty/whitespace doc_names. Such rows (e.g. from PDFs
        # whose title metadata was empty) would otherwise produce a hidden
        # "sources/.md" page that nothing lists. Replace with a visible name.
        fixed: dict[str, str] = {}
        empty_count = 0
        for name, content in texts.items():
            if not (name or "").strip():
                name = "untitled_document"
                empty_count += 1
            fixed[name] = content
        if empty_count:
            logger.warning(
                f"{empty_count} document group(s) in the embedding table have an "
                f"empty doc_name and will be combined into a single "
                f"'untitled_document' page. Re-embed the sources (or run "
                f"DocEmbedder.backfill_empty_doc_names) to name them properly."
            )
        texts = fixed

        if merge_parts:
            raw_count = len(texts)
            texts = self._group_documents_by_base(texts)
            if len(texts) < raw_count:
                logger.info(
                    f"Merged {raw_count} embedded parts into {len(texts)} "
                    f"document(s) before wiki ingest"
                )

        results = []
        errors = []
        total_pages = 0
        for name, content in texts.items():
            if not content.strip():
                logger.info(f"Skipping empty document '{name}'")
                continue
            logger.info(f"Ingesting embedded document into wiki: {name}")
            try:
                result = self.ingest_source(name, content)
            except Exception as e:
                # One bad document must not abort the whole batch.
                logger.exception(f"Failed to ingest '{name}' into wiki: {e}")
                errors.append({"source": name, "error": str(e)})
                continue
            results.append(result)
            total_pages += result.get("pages_touched", 0)

        logger.success(
            f"Wiki ingest from embeddings complete: {len(results)} sources, "
            f"{total_pages} pages touched"
            + (f", {len(errors)} error(s)" if errors else "")
        )
        return {
            "collection": collection,
            "documents_ingested": len(results),
            "pages_touched": total_pages,
            "results": results,
            "errors": errors,
        }

    def _structured_llm_call(self, prompt: str, response_model, context: str = ""):
        """
        Call the LLM with structured output, falling back to plain text + JSON extraction.

        Step 1: Try pydantic-ai's native structured output via self._struct_llm.
        Step 2: If unsupported, use the plain-text agent (self._llm) with schema hints
                and extract JSON from the response.
        """
        self._clear_chat_history()

        # Step 1: Try structured output (only if not known to be unsupported)
        if self._structured_output_supported is not False:
            try:
                result = self._struct_llm.get_response(
                    question=prompt,
                    context=context,
                    response_model=response_model,
                )
                if isinstance(result, response_model):
                    self._structured_output_supported = True
                    return result
                self._structured_output_supported = False
                logger.warning(
                    f"Structured output returned {type(result).__name__}, "
                    f"falling back to plain-text agent"
                )
            except Exception as e:
                self._structured_output_supported = False
                logger.warning(f"Structured output failed ({e}), falling back to plain-text agent")

        # Step 2: Plain-text agent with schema hint + JSON extraction
        self._clear_chat_history()
        schema_hint = self._build_schema_hint(response_model)
        json_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a valid JSON object using exactly these fields:\n"
            f"{schema_hint}\n\n"
            f"No prose, no explanation, no markdown code blocks. Just the JSON object."
        )
        try:
            raw = self._llm.get_response(question=json_prompt, context=context)
            parsed = self._extract_json_from_response(raw, response_model)
            if parsed is not None:
                logger.info("Successfully extracted JSON from plain-text agent")
                return parsed
            logger.error("JSON extraction failed from plain-text agent response")
        except Exception as e:
            logger.error(f"Plain-text agent also failed: {e}")
        return None

    def _clear_chat_history(self):
        """Clear both LLM agents' chat histories to avoid stale messages."""
        self._struct_llm.chat_history.queue.clear()
        self._llm.chat_history.queue.clear()

    @staticmethod
    def _build_schema_hint(model) -> str:
        """Build a concise field description from a Pydantic model for LLM prompts."""
        hints = []
        for name, field_info in model.model_fields.items():
            ann = field_info.annotation
            desc = field_info.description or ""
            required = field_info.is_required()
            req_str = "required" if required else "optional"

            # Check if the annotation is a list of Pydantic models
            origin = getattr(ann, "__origin__", None)
            if origin is list:
                args = getattr(ann, "__args__", ())
                if args and hasattr(args[0], "model_fields"):
                    # Nested Pydantic model — show its fields
                    sub_hints = []
                    for sub_name, sub_field in args[0].model_fields.items():
                        sub_ann = sub_field.annotation
                        sub_type = getattr(sub_ann, "__name__", str(sub_ann))
                        sub_req = "required" if sub_field.is_required() else "optional"
                        sub_desc = sub_field.description or ""
                        sub_hints.append(f'      "{sub_name}": {sub_type} ({sub_req}) {sub_desc}')
                    hints.append(
                        f'  "{name}": list of objects with fields:\n'
                        + "\n".join(sub_hints)
                    )
                    continue

                type_str = f"list of {getattr(args[0], '__name__', str(args[0])) if args else 'any'}"
            elif hasattr(ann, "model_fields"):
                sub_hints = []
                for sub_name, sub_field in ann.model_fields.items():
                    sub_type = getattr(sub_field.annotation, "__name__", str(sub_field.annotation))
                    sub_req = "required" if sub_field.is_required() else "optional"
                    sub_hints.append(f'      "{sub_name}": {sub_type} ({sub_req})')
                hints.append(
                    f'  "{name}": object with fields:\n' + "\n".join(sub_hints)
                )
                continue
            else:
                type_str = getattr(ann, "__name__", str(ann))

            hints.append(f'  "{name}": {type_str} ({req_str}) — {desc}')
        return "\n".join(hints)

    def _extract_json_from_response(self, response, response_model):
        """Try to extract and validate JSON from an LLM text response.

        When the model wraps the JSON in reasoning, there may be several
        ``{...}`` fragments in the text (including nested entity/concept
        objects). We collect every parseable object and pick the one whose
        top-level keys overlap most with ``response_model``'s fields — that
        is the real outer payload, not a nested sub-object (which would
        otherwise coerce to an empty model with all-default fields).
        """
        import json as _json

        raw = response if isinstance(response, str) else str(response)
        text = raw.strip()

        model_fields = set(response_model.model_fields.keys())

        def _score(obj) -> int:
            """Number of model-field keys present at the top level."""
            return len(model_fields & set(obj.keys())) if isinstance(obj, dict) else -1

        best_obj = None
        best_score = 0

        def _consider(obj) -> bool:
            nonlocal best_obj, best_score
            s = _score(obj)
            if s > best_score:
                best_score = s
                best_obj = obj
                return True
            return False

        # Markdown code block (most reliable when present).
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                try:
                    _consider(_json.loads(match.group(1).strip()))
                except _json.JSONDecodeError:
                    pass

        # Whole text if it starts with {.
        if text.startswith("{"):
            try:
                _consider(_json.loads(text))
            except _json.JSONDecodeError:
                pass

        # Scan every brace position for a valid JSON object.
        decoder = _json.JSONDecoder()
        brace_positions = [i for i, ch in enumerate(text) if ch == "{"]
        for i in brace_positions:
            try:
                obj, _end = decoder.raw_decode(text, i)
            except _json.JSONDecodeError:
                continue
            _consider(obj)

        if best_obj is not None:
            parsed = self._try_parse_json(_json.dumps(best_obj), response_model, _json)
            if parsed is not None:
                return parsed

        return None

    def _try_parse_json(self, json_str: str, response_model, _json):
        """Try to parse a JSON string into a Pydantic model, with coercion."""
        try:
            return response_model.model_validate_json(json_str)
        except Exception:
            pass
        try:
            data = _json.loads(json_str)
            data = self._coerce_to_schema(data, response_model)
            return response_model.model_validate(data)
        except Exception:
            return None

    @staticmethod
    def _coerce_to_schema(data: dict, model) -> dict:
        """
        Coerce raw LLM JSON data to match a Pydantic model's expected shape.
        Handles common mismatches: wrong field names, missing required fields,
        string values where objects are expected.
        """
        if not isinstance(data, dict):
            return data

        coerced = {}
        for name, field_info in model.model_fields.items():
            ann = field_info.annotation
            origin = getattr(ann, "__origin__", None)

            if name in data:
                value = data[name]
            elif name == "entity_type" and "type" in data:
                value = data["type"]
            elif not field_info.is_required():
                continue
            else:
                # Provide defaults for missing required string fields
                coerced[name] = ""
                continue

            # Handle list of Pydantic submodels
            if origin is list:
                args = getattr(ann, "__args__", ())
                if args and hasattr(args[0], "model_fields") and isinstance(value, list):
                    sub_model = args[0]
                    coerced[name] = [
                        WikiManager._coerce_to_schema(
                            item if isinstance(item, dict) else {"name": str(item)},
                            sub_model,
                        )
                        for item in value
                    ]
                    continue

            # Handle Pydantic submodel (single object)
            if hasattr(ann, "model_fields") and isinstance(value, dict):
                coerced[name] = WikiManager._coerce_to_schema(value, ann)
                continue

            # Handle string where list expected
            if origin is list and isinstance(value, str):
                coerced[name] = [value]
                continue

            coerced[name] = value

        return coerced

    def _generate_source_summary(self, doc_name: str, doc_content: str) -> SourceSummary:
        """Use the LLM to generate a structured summary of a source."""
        prompt = (
            f"You are a disciplined wiki maintainer. Read the source below and produce "
            f"a structured summary. Extract entities, concepts, and flag any claims that "
            f"might contradict common knowledge or previously established facts.\n\n"
            f"Source: {doc_name}\n\n{doc_content[:12000]}"
        )
        result = self._structured_llm_call(prompt, SourceSummary)
        if result is not None:
            return result
        logger.error("All attempts to generate source summary failed, using fallback")
        return SourceSummary(
            title=doc_name,
            summary="(Summary generation failed)",
            key_takeaways=[],
            entities=[],
            concepts=[],
            contradictions=[],
            questions_raised=[],
        )

    def _generate_update_plan(self, doc_name: str, summary: SourceSummary) -> WikiUpdatePlan:
        """Use the LLM to plan which wiki pages need updates."""
        index_content = self._read_page(self.index_path)

        prompt = (
            f"Based on the following source summary, plan which wiki pages need to be "
            f"created or updated. Consider the existing wiki index below.\n\n"
            f"Source: {doc_name}\n"
            f"Summary: {summary.summary}\n"
            f"Entities: {[e.name for e in summary.entities]}\n"
            f"Concepts: {[c.name for c in summary.concepts]}\n\n"
            f"Existing Wiki Index:\n{index_content[:4000]}"
        )
        result = self._structured_llm_call(prompt, WikiUpdatePlan)
        if result is not None:
            return result
        logger.error("All attempts to generate update plan failed, using fallback")
        return WikiUpdatePlan(
            source_title=doc_name,
            pages_to_update=[],
            pages_to_link=[],
            synthesis_notes="",
        )

    def _build_source_page(
        self, doc_name: str, summary: SourceSummary, source_type: str
    ) -> str:
        """Build markdown content for a source summary page."""
        date_str = datetime.now().isoformat()
        takeaways = "\n".join(f"- {t}" for t in summary.key_takeaways) or "_None extracted._"
        entities = "\n".join(
            f"- [[{e.name}]] — {e.description[:100]}" for e in summary.entities
        ) or "_None extracted._"
        concepts = "\n".join(
            f"- [[{c.name}]] — {c.description[:100]}" for c in summary.concepts
        ) or "_None extracted._"
        questions = "\n".join(f"- {q}" for q in summary.questions_raised) or "_None raised._"

        return SOURCE_PAGE_TEMPLATE.format(
            title=summary.title or doc_name,
            date=date_str,
            source_type=source_type,
            summary=summary.summary,
            takeaways=takeaways,
            entities=entities,
            concepts=concepts,
            questions=questions,
            collection=self.collection_name,
        )

    def _update_entity_page(self, entity, doc_name: str) -> int:
        """Create or append to an entity page. Returns 1 if a page was written."""
        page_name = self._sanitize_name(entity.name)
        page_path = self.entities_dir / f"{page_name}.md"
        date_str = datetime.now().isoformat()

        if page_path.exists():
            content = self._read_page(page_path)
            fm, body = self._parse_frontmatter(content)
            # Append mention
            mention_line = f"- [[{self._sanitize_name(doc_name)}|{doc_name}]]"
            if mention_line not in body:
                body += f"\n{mention_line}"
            # Update related entities if new ones found
            existing_related = self._extract_wikilinks(body)
            for rel in entity.related_entities:
                if rel not in existing_related:
                    body += f"\n- [[{rel}]]"
            content = self._build_frontmatter(fm) + body
        else:
            related = "\n".join(f"- [[{r}]]" for r in entity.related_entities) or "_None yet._"
            mentions = f"- [[{self._sanitize_name(doc_name)}|{doc_name}]]"
            content = ENTITY_PAGE_TEMPLATE.format(
                title=entity.name,
                entity_type=entity.entity_type,
                date=date_str,
                description=entity.description,
                mentions=mentions,
                related=related,
            )

        self._write_page(page_path, content)
        return 1

    def _update_concept_page(self, concept, doc_name: str) -> int:
        """Create or append to a concept page. Returns 1 if a page was written."""
        page_name = self._sanitize_name(concept.name)
        page_path = self.concepts_dir / f"{page_name}.md"
        date_str = datetime.now().isoformat()

        if page_path.exists():
            content = self._read_page(page_path)
            fm, body = self._parse_frontmatter(content)
            mention_line = f"- [[{self._sanitize_name(doc_name)}|{doc_name}]]"
            if mention_line not in body:
                body += f"\n{mention_line}"
            existing_related = self._extract_wikilinks(body)
            for rel in concept.related_concepts:
                if rel not in existing_related:
                    body += f"\n- [[{rel}]]"
            content = self._build_frontmatter(fm) + body
        else:
            related = "\n".join(f"- [[{r}]]" for r in concept.related_concepts) or "_None yet._"
            mentions = f"- [[{self._sanitize_name(doc_name)}|{doc_name}]]"
            content = CONCEPT_PAGE_TEMPLATE.format(
                title=concept.name,
                date=date_str,
                description=concept.description,
                mentions=mentions,
                related=related,
            )

        self._write_page(page_path, content)
        return 1

    def _update_synthesis(self, notes: str, doc_name: str) -> int:
        """Append synthesis notes to a running synthesis page. Returns 1."""
        synthesis_path = self.synthesis_dir / "overview.md"
        date_str = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n\n## [{date_str}] From [[{self._sanitize_name(doc_name)}|{doc_name}]]\n\n{notes}"

        if synthesis_path.exists():
            content = self._read_page(synthesis_path)
        else:
            content = "# Synthesis Overview\n\n> Evolving synthesis of all sources in this collection.\n"

        content += entry
        self._write_page(synthesis_path, content)
        return 1

    # ────────────────────────── consolidate ────────────────────────

    def consolidate_part_pages(self) -> dict[str, Any]:
        """
        Merge per-part source pages into one page per original document.

        Scans ``sources/`` for pages whose stem carries a part suffix (e.g.
        ``report_part1``, ``report_part2``, ``report (1)``) and merges each
        group into a single page named after the base document (``report``).
        Wikilinks across the whole wiki that pointed at the part pages are
        rewritten to the merged page.

        This is a pure filesystem operation — no LLM or embedder is required.
        To regenerate fresh LLM summaries for the merged documents, re-run
        :meth:`ingest_from_embeddings` afterwards.

        :returns: stats dict with ``groups_merged``, ``pages_removed`` and
            ``links_rewritten`` counts.
        """
        logger.info("Consolidating per-part wiki source pages...")

        source_files: dict[str, Path] = {
            p.stem: p for p in self.sources_dir.glob("*.md")
        }

        # Group source page stems by their base document name.
        buckets: dict[str, list[tuple[int | None, str, Path]]] = {}
        for stem, path in source_files.items():
            base, part = self._doc_base_and_part(stem)
            buckets.setdefault(base, []).append((part, stem, path))

        groups_merged = 0
        pages_removed = 0
        links_rewritten = 0

        for base, items in buckets.items():
            part_items = [(p, s, path) for p, s, path in items if p is not None]
            non_part_items = [(p, s, path) for p, s, path in items if p is None]
            if not part_items:
                # No part-suffixed page in this group → nothing to consolidate.
                continue

            # Order: any existing base page first, then parts by index.
            part_items.sort(
                key=lambda t: (t[0] if t[0] is not None else float("inf"), t[1])
            )
            ordered = non_part_items + part_items

            base_stem = self._sanitize_name(base)
            base_path = self.sources_dir / f"{base_stem}.md"

            merged_md = self._merge_source_markdown(base_stem, ordered)
            self._write_page(base_path, merged_md)

            # Remove the now-merged source files (parts + old base page).
            for _, stem, path in ordered:
                if path.resolve() == base_path.resolve():
                    continue
                if path.exists():
                    path.unlink()
                    pages_removed += 1

            # Rewrite links that pointed at the part pages.
            links_rewritten += self._rewrite_wikilinks(
                [stem for _, stem, _ in part_items], base_stem
            )
            groups_merged += 1

        if groups_merged:
            self._write_index()
            self._append_log(
                "consolidate",
                f"merged {groups_merged} document group(s)",
                pages_removed,
            )

        logger.success(
            f"Consolidation complete: {groups_merged} group(s) merged, "
            f"{pages_removed} part page(s) removed, "
            f"{links_rewritten} link(s) rewritten"
        )
        return {
            "groups_merged": groups_merged,
            "pages_removed": pages_removed,
            "links_rewritten": links_rewritten,
        }

    def _merge_source_markdown(
        self, base_stem: str, parts: list[tuple[int | None, str, Path]]
    ) -> str:
        """
        Build merged markdown for several source pages under one base page.

        :param base_stem: sanitized stem to use as the merged page filename.
        :param parts: ordered list of ``(part_index, stem, path)``.
        """
        date_str = datetime.now().isoformat()

        # Recover a clean display title from the first available frontmatter.
        display_title = base_stem
        for _part, stem, path in parts:
            content = self._read_page(path)
            fm, _ = self._parse_frontmatter(content)
            fm_title = str(fm.get("title") or stem)
            clean_title, _ = self._doc_base_and_part(fm_title)
            if clean_title:
                display_title = clean_title
                break

        multi = len(parts) > 1
        sections: list[str] = []
        for part, _stem, path in parts:
            content = self._read_page(path)
            _, body = self._parse_frontmatter(content)
            if not body.strip():
                continue
            # Drop the leading "# Title" line; we provide our own headers.
            body_lines = body.split("\n")
            if body_lines and body_lines[0].lstrip().startswith("# "):
                body_lines = body_lines[1:]
            body = "\n".join(body_lines).strip()
            if not body:
                continue
            if multi:
                label = f"Part {part}" if part is not None else _stem
                sections.append(f"## {label}\n\n{body}")
            else:
                sections.append(body)

        body = "\n\n".join(sections).strip()
        fm = {
            "title": display_title,
            "date_ingested": date_str,
            "source_type": "document",
        }
        if multi:
            fm["merged_from"] = [stem for _, stem, _ in parts]
        return self._build_frontmatter(fm) + f"# {display_title}\n\n{body}\n"

    def _rewrite_wikilinks(self, old_stems: list[str], new_stem: str) -> int:
        """
        Rewrite ``[[old_stem...]]`` wikilinks anywhere in the wiki to
        ``[[new_stem...]]``, preserving any alias (``|alias``) or anchor
        (``#anchor``) suffix. Returns the number of links rewritten.
        """
        if not old_stems:
            return 0
        pattern = re.compile(
            r"\[\[("
            + "|".join(re.escape(s) for s in old_stems)
            + r")((?:#[^\]]*)?(?:\|[^]]*)?)\]\]",
            re.IGNORECASE,
        )
        count = 0
        for md_file in self.wiki_dir.rglob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            content = self._read_page(md_file)
            new_content, n = pattern.subn(
                lambda m: f"[[{new_stem}{m.group(2)}]]", content
            )
            if n:
                self._write_page(md_file, new_content)
                count += n
        return count

    # ────────────────────────── query ───────────────────────────────

    def query(self, question: str, file_answer: bool = False) -> dict[str, Any]:
        """
        Answer a question using the wiki.

        Steps:
        1. Read index.md to find relevant pages.
        2. Read the most relevant pages into context.
        3. Synthesize an answer with citations.
        4. Optionally file the answer back into the wiki.
        """
        logger.info(f"Wiki query: {question}")

        index_content = self._read_page(self.index_path)

        # Gather relevant pages (simple heuristic: all non-index/log pages for now)
        page_contents: list[tuple[str, str]] = []
        for md_file in self.wiki_dir.rglob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            rel = md_file.relative_to(self.wiki_dir).as_posix()
            page_contents.append((rel, self._read_page(md_file)))

        # Limit context to avoid token overflow
        max_pages = 15
        ranked_paths = self._graph_rank_pages(question)
        if ranked_paths:
            # Graph-guided selection: ranked pages first, remaining pages appended
            by_rel = dict(page_contents)
            ranked = [(rel, by_rel[rel]) for rel in ranked_paths if rel in by_rel]
            seen = {rel for rel, _ in ranked}
            remaining = [(rel, content) for rel, content in page_contents if rel not in seen]
            page_contents = (ranked + remaining)[:max_pages]
        elif len(page_contents) > max_pages:
            # Fallback: simple keyword relevance filter
            keywords = set(question.lower().split())
            scored = []
            for rel, content in page_contents:
                score = sum(1 for kw in keywords if kw in content.lower())
                scored.append((score, rel, content))
            scored.sort(reverse=True)
            page_contents = [(rel, content) for _, rel, content in scored[:max_pages]]

        context = self._build_query_context(index_content, page_contents)

        answer = self._generate_answer(question, context)

        result = {
            "question": question,
            "answer": answer.answer,
            "sources_used": answer.sources_used,
            "confidence": answer.confidence,
            "gaps": answer.gaps,
            "suggested_followups": answer.suggested_followups,
        }

        if file_answer:
            page_title = self._sanitize_name(question[:60])
            answer_path = self.synthesis_dir / f"{page_title}.md"
            answer_content = (
                f"---\n"
                f"title: Answer — {question[:80]}\n"
                f"date_created: {datetime.now().isoformat()}\n"
                f"query: {question}\n"
                f"confidence: {answer.confidence}\n"
                f"---\n\n"
                f"# {question}\n\n"
                f"{answer.answer}\n\n"
                f"## Sources Used\n\n"
                + "\n".join(f"- [[{s}]]" for s in answer.sources_used)
                + "\n\n"
                + "## Gaps\n\n"
                + "\n".join(f"- {g}" for g in answer.gaps)
                + "\n\n"
                + "## Suggested Follow-ups\n\n"
                + "\n".join(f"- {f}" for f in answer.suggested_followups)
                + "\n"
            )
            self._write_page(answer_path, answer_content)
            self._write_index()
            self._append_log("query", f"filed answer for: {question[:60]}", 1)
            result["filed_page"] = answer_path.name

        return result

    def _build_query_context(
        self, index_content: str, page_contents: list[tuple[str, str]]
    ) -> str:
        """Build the context string for a query from relevant pages."""
        parts = ["# Wiki Index\n\n" + index_content]
        for rel, content in page_contents:
            parts.append(f"\n---\n\n# Page: {rel}\n\n{content}")
        return "\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> WikiQueryAnswer:
        """Use the LLM to synthesize an answer from wiki context."""
        prompt = (
            f"You are a research assistant answering questions from a personal knowledge wiki. "
            f"Use ONLY the provided wiki pages to answer. Cite pages inline using [[Page Name]] syntax. "
            f"If the wiki lacks sufficient information, say so clearly and suggest what sources to look for.\n\n"
            f"Question: {question}\n\n"
            f"Wiki Context:\n{context[:20000]}"
        )
        result = self._structured_llm_call(prompt, WikiQueryAnswer)
        if result is not None:
            return result
        logger.error("All attempts to generate wiki answer failed, using fallback")
        return WikiQueryAnswer(
                answer="(Answer generation failed)",
                sources_used=[],
                confidence="low",
                gaps=["LLM error occurred"],
                suggested_followups=[],
            )

    # ────────────────────────── lint ────────────────────────────────

    def lint(self, auto_fix: bool = False) -> dict[str, Any]:
        """
        Health-check the wiki.

        Detects: orphans, broken links, contradictions, stale claims, missing pages.
        Optionally applies simple fixes (stub creation, redirects).
        """
        logger.info("Running wiki lint...")

        orphans = self._find_orphans()
        broken = self._find_broken_links()

        # Read all page contents for deeper analysis
        all_pages: dict[str, str] = {}
        for md_file in self.wiki_dir.rglob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            rel = md_file.relative_to(self.wiki_dir).as_posix()
            all_pages[rel] = self._read_page(md_file)

        # Use LLM for contradictions, stale claims, and missing pages
        report = self._generate_lint_report(all_pages)

        # Add structural issues
        report.orphan_pages = orphans
        report.broken_links = broken

        fixes_applied = 0
        if auto_fix:
            for link in broken:
                # Create a stub page for broken links
                stub_path = self.concepts_dir / f"{self._sanitize_name(link)}.md"
                if not stub_path.exists():
                    stub_content = (
                        f"---\n"
                        f"title: {link}\n"
                        f"date_created: {datetime.now().isoformat()}\n"
                        f"status: stub\n"
                        f"---\n\n"
                        f"# {link}\n\n"
                        f"> This page was auto-created because it is referenced elsewhere in the wiki.\n\n"
                        f"_Stub — needs content._\n"
                    )
                    self._write_page(stub_path, stub_content)
                    fixes_applied += 1

            if fixes_applied:
                self._write_index()
                self._append_log("lint", f"auto-fix applied: created {fixes_applied} stubs", fixes_applied)

        self._append_log("lint", f"found {len(orphans)} orphans, {len(broken)} broken links", 0)

        result = {
            "orphan_pages": report.orphan_pages,
            "broken_links": report.broken_links,
            "contradictions": [c.model_dump() for c in report.contradictions],
            "stale_claims": [s.model_dump() for s in report.stale_claims],
            "missing_pages": [m.model_dump() for m in report.missing_pages],
            "suggestions": report.suggestions,
            "fixes_applied": fixes_applied,
        }

        if self.graph_enabled:
            try:
                result["graph_hubs"] = self._get_knowledge_graph().hubs(5)
            except Exception as e:
                logger.warning(f"Could not compute graph hubs for lint: {e}")
                result["graph_hubs"] = []

        return result

    def _generate_lint_report(self, all_pages: dict[str, str]) -> LintReport:
        """Use the LLM to analyze pages for contradictions, stale claims, and gaps."""
        # Summarize pages for the LLM to keep token count manageable
        summaries = []
        for rel, content in all_pages.items():
            fm, body = self._parse_frontmatter(content)
            title = fm.get("title", Path(rel).stem)
            first_para = body.split("\n\n")[0][:300]
            summaries.append(f"Page: {rel} (title: {title})\n{first_para}\n")

        prompt = (
            f"You are a wiki editor doing a health check. Review the following page summaries "
            f"and identify contradictions, stale claims, missing concept pages, and general improvements.\n\n"
            f"{'\n'.join(summaries)[:20000]}"
        )
        result = self._structured_llm_call(prompt, LintReport)
        if result is not None:
            return result
        logger.error("All attempts to generate lint report failed, using fallback")
        return LintReport()

    # ────────────────────────── status ──────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return statistics about the wiki."""
        page_counts = {
            "sources": len(list(self.sources_dir.glob("*.md"))),
            "entities": len(list(self.entities_dir.glob("*.md"))),
            "concepts": len(list(self.concepts_dir.glob("*.md"))),
            "synthesis": len(list(self.synthesis_dir.glob("*.md"))),
        }
        total = sum(page_counts.values())
        orphans = self._find_orphans()
        broken = self._find_broken_links()

        last_log = ""
        if self.log_path.exists():
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            log_lines = [l for l in lines if l.startswith("## [")]
            if log_lines:
                last_log = log_lines[-1]

        result = {
            "collection": self.collection_name,
            "wiki_path": str(self.wiki_dir),
            "total_pages": total,
            "page_counts": page_counts,
            "orphan_pages": len(orphans),
            "broken_links": len(broken),
            "last_operation": last_log,
        }

        if self.graph_enabled and self._get_knowledge_graph().graph_path.exists():
            try:
                result["graph"] = self.graph_status()
            except Exception as e:
                logger.warning(f"Could not read graph status: {e}")
                result["graph"] = None
        else:
            result["graph"] = None

        return result
