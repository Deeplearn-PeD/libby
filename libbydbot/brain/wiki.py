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
from datetime import datetime
from pathlib import Path
from typing import Any

import loguru
import yaml

from libbydbot.brain.wiki_models import (
    LintReport,
    SourceSummary,
    WikiQueryAnswer,
    WikiUpdatePlan,
)
from base_agent.llminterface import StructuredLangModel

logger = loguru.logger

DEFAULT_WIKI_BASE = Path.home() / ".libby" / "wikis"

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

    def __init__(
        self,
        collection_name: str,
        wiki_base: str | Path = "",
        model: str = "llama3.2",
    ):
        self.collection_name = collection_name
        self.wiki_base = Path(wiki_base) if wiki_base else DEFAULT_WIKI_BASE
        self.wiki_dir = self.wiki_base / self._sanitize_name(collection_name)
        self.model = model
        self._llm = StructuredLangModel(model=model)
        self._ensure_structure()

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

        logger.success(f"Wiki ingest complete: {doc_name} ({pages_touched} pages touched)")
        return {
            "source": doc_name,
            "pages_touched": pages_touched,
            "entities_created": len(summary.entities),
            "concepts_created": len(summary.concepts),
            "summary": summary.summary,
        }

    def _generate_source_summary(self, doc_name: str, doc_content: str) -> SourceSummary:
        """Use the LLM to generate a structured summary of a source."""
        prompt = (
            f"You are a disciplined wiki maintainer. Read the source below and produce "
            f"a structured summary. Extract entities, concepts, and flag any claims that "
            f"might contradict common knowledge or previously established facts.\n\n"
            f"Source: {doc_name}\n\n{doc_content[:12000]}"
        )
        try:
            result = self._llm.get_response(
                question=prompt,
                context="",
                response_model=SourceSummary,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to generate source summary: {e}")
            # Return a minimal fallback summary
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
        # Read existing index to give the LLM context
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
        try:
            result = self._llm.get_response(
                question=prompt,
                context="",
                response_model=WikiUpdatePlan,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to generate update plan: {e}")
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
        # Future: use embeddings or keyword matching to select pages
        page_contents: list[tuple[str, str]] = []
        for md_file in self.wiki_dir.rglob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            rel = md_file.relative_to(self.wiki_dir).as_posix()
            page_contents.append((rel, self._read_page(md_file)))

        # Limit context to avoid token overflow
        max_pages = 15
        if len(page_contents) > max_pages:
            # Simple keyword relevance filter
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
        try:
            result = self._llm.get_response(
                question=prompt,
                context="",
                response_model=WikiQueryAnswer,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to generate wiki query answer: {e}")
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

        return {
            "orphan_pages": report.orphan_pages,
            "broken_links": report.broken_links,
            "contradictions": [c.model_dump() for c in report.contradictions],
            "stale_claims": [s.model_dump() for s in report.stale_claims],
            "missing_pages": [m.model_dump() for m in report.missing_pages],
            "suggestions": report.suggestions,
            "fixes_applied": fixes_applied,
        }

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
        try:
            result = self._llm.get_response(
                question=prompt,
                context="",
                response_model=LintReport,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to generate lint report: {e}")
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

        return {
            "collection": self.collection_name,
            "wiki_path": str(self.wiki_dir),
            "total_pages": total,
            "page_counts": page_counts,
            "orphan_pages": len(orphans),
            "broken_links": len(broken),
            "last_operation": last_log,
        }
