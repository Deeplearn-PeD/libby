import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libbydbot.brain.wiki import WikiManager
from libbydbot.brain.wiki_models import (
    KeyConcept,
    KeyEntity,
    LintReport,
    SourceSummary,
    WikiQueryAnswer,
    WikiUpdatePlan,
)


@pytest.fixture
def temp_wiki():
    """Create a temporary wiki directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wiki = WikiManager(
            collection_name="test_collection",
            wiki_base=tmpdir,
            model="llama3.2",
        )
        # Replace the LLM with a mock to avoid real API calls
        wiki._llm = MagicMock()
        yield wiki


class TestWikiStructure:
    def test_directory_creation(self, temp_wiki):
        assert temp_wiki.wiki_dir.exists()
        assert temp_wiki.sources_dir.exists()
        assert temp_wiki.entities_dir.exists()
        assert temp_wiki.concepts_dir.exists()
        assert temp_wiki.synthesis_dir.exists()

    def test_index_created(self, temp_wiki):
        assert temp_wiki.index_path.exists()
        content = temp_wiki.index_path.read_text(encoding="utf-8")
        assert "Wiki Index" in content

    def test_log_created(self, temp_wiki):
        assert temp_wiki.log_path.exists()
        content = temp_wiki.log_path.read_text(encoding="utf-8")
        assert "Wiki Log" in content


class TestPageIO:
    def test_write_and_read_page(self, temp_wiki):
        path = temp_wiki.wiki_dir / "test_page.md"
        temp_wiki._write_page(path, "# Hello\n\nWorld")
        assert path.exists()
        assert temp_wiki._read_page(path) == "# Hello\n\nWorld"

    def test_parse_frontmatter(self, temp_wiki):
        content = "---\ntitle: Test\n---\n\n# Body\n"
        fm, body = temp_wiki._parse_frontmatter(content)
        assert fm["title"] == "Test"
        assert "# Body" in body

    def test_build_frontmatter(self, temp_wiki):
        fm = {"title": "Test", "date": "2026-04-21"}
        result = temp_wiki._build_frontmatter(fm)
        assert result.startswith("---")
        assert "title: Test" in result


class TestLinkGraph:
    def test_extract_wikilinks(self, temp_wiki):
        content = "See [[Entity A]] and [[Concept B]] for details."
        links = temp_wiki._extract_wikilinks(content)
        assert links == ["Entity A", "Concept B"]

    def test_find_orphans(self, temp_wiki):
        # Create two pages, one linking to the other
        temp_wiki._write_page(temp_wiki.entities_dir / "alice.md", "# Alice\n\n[[Bob]]")
        temp_wiki._write_page(temp_wiki.entities_dir / "bob.md", "# Bob\n\nAlice's friend.")
        orphans = temp_wiki._find_orphans()
        # alice.md has no inbound links (bob does not link back to alice)
        assert "entities/alice.md" in orphans
        # bob.md is linked from alice.md, so it is not an orphan
        assert "entities/bob.md" not in orphans

    def test_find_broken_links(self, temp_wiki):
        temp_wiki._write_page(
            temp_wiki.entities_dir / "alice.md", "# Alice\n\n[[NonExistent]]"
        )
        broken = temp_wiki._find_broken_links()
        assert "NonExistent" in broken


class TestLogAndIndex:
    def test_append_log(self, temp_wiki):
        temp_wiki._append_log("ingest", "Test Doc", 3)
        log_content = temp_wiki.log_path.read_text(encoding="utf-8")
        assert "ingest | Test Doc" in log_content
        assert "touched 3 pages" in log_content

    def test_write_index_lists_pages(self, temp_wiki):
        temp_wiki._write_page(temp_wiki.sources_dir / "doc_a.md", "# Doc A\n\nSummary of doc A.")
        temp_wiki._write_index()
        index_content = temp_wiki.index_path.read_text(encoding="utf-8")
        assert "Doc A" in index_content


class TestIngest:
    def test_ingest_source_creates_pages(self, temp_wiki):
        # Mock LLM responses
        temp_wiki._generate_source_summary = MagicMock(
            return_value=SourceSummary(
                title="Test Doc",
                summary="A test document.",
                key_takeaways=["Point 1"],
                entities=[
                    KeyEntity(
                        name="Alice",
                        entity_type="person",
                        description="A person.",
                        related_entities=[],
                    )
                ],
                concepts=[
                    KeyConcept(
                        name="Testing",
                        description="The art of testing.",
                        related_concepts=[],
                    )
                ],
                contradictions=[],
                questions_raised=[],
            )
        )
        temp_wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(
                source_title="Test Doc",
                pages_to_update=[],
                pages_to_link=[],
                synthesis_notes="",
            )
        )

        result = temp_wiki.ingest_source("Test Doc", "This is the content of the test document.")

        assert result["source"] == "Test Doc"
        assert result["pages_touched"] >= 1
        assert (temp_wiki.sources_dir / "test_doc.md").exists()

    def test_ingest_from_embeddings(self, tmp_path):
        """Build the wiki straight from the embedding table (no PDF re-parse)."""
        import numpy as np
        from libbydbot.brain.embed import DocEmbedder

        with patch(
            "libbydbot.brain.embed.DocEmbedder._generate_embedding"
        ) as mocked, patch(
            "libbydbot.brain.embed.DocEmbedder._get_embedding_dimension",
            return_value=1024,
        ):
            mocked.return_value = np.zeros(1024).tolist()
            db_path = tmp_path / "emb.db"
            embedder = DocEmbedder(
                "test_collection",
                dburl=f"sqlite:///{db_path}",
                embedding_model="mxbai-embed-large",
            )
            embedder.embed_text("Alpha page one.", "alpha_doc", 0)
            embedder.embed_text("Alpha page two.", "alpha_doc", 1)
            embedder.embed_text("Beta standalone.", "beta_doc", 0)

        wiki = WikiManager(
            collection_name="test_collection",
            wiki_base=str(tmp_path / "wiki"),
            model="llama3.2",
        )
        wiki._generate_source_summary = MagicMock(
            return_value=SourceSummary(
                title="x",
                summary="s",
                key_takeaways=[],
                entities=[],
                concepts=[],
                contradictions=[],
                questions_raised=[],
            )
        )
        wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(
                source_title="x",
                pages_to_update=[],
                pages_to_link=[],
                synthesis_notes="",
            )
        )

        result = wiki.ingest_from_embeddings(embedder, collection="test_collection")

        assert result["documents_ingested"] == 2
        assert result["pages_touched"] >= 2
        assert (wiki.sources_dir / "alpha_doc.md").exists()
        assert (wiki.sources_dir / "beta_doc.md").exists()

    def test_ingest_updates_entity_page(self, temp_wiki):
        # First ingest
        temp_wiki._generate_source_summary = MagicMock(
            return_value=SourceSummary(
                title="Doc 1",
                summary="First doc.",
                key_takeaways=[],
                entities=[
                    KeyEntity(
                        name="Alice",
                        entity_type="person",
                        description="Alice from doc 1.",
                        related_entities=[],
                    )
                ],
                concepts=[],
                contradictions=[],
                questions_raised=[],
            )
        )
        temp_wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(
                source_title="Doc 1",
                pages_to_update=[],
                pages_to_link=[],
                synthesis_notes="",
            )
        )
        temp_wiki.ingest_source("Doc 1", "Content 1")

        # Second ingest with same entity
        temp_wiki._generate_source_summary = MagicMock(
            return_value=SourceSummary(
                title="Doc 2",
                summary="Second doc.",
                key_takeaways=[],
                entities=[
                    KeyEntity(
                        name="Alice",
                        entity_type="person",
                        description="Alice from doc 2.",
                        related_entities=[],
                    )
                ],
                concepts=[],
                contradictions=[],
                questions_raised=[],
            )
        )
        temp_wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(
                source_title="Doc 2",
                pages_to_update=[],
                pages_to_link=[],
                synthesis_notes="",
            )
        )
        temp_wiki.ingest_source("Doc 2", "Content 2")

        entity_page = temp_wiki.entities_dir / "alice.md"
        content = entity_page.read_text(encoding="utf-8")
        assert "doc_1" in content or "Doc 1" in content
        assert "doc_2" in content or "Doc 2" in content


class TestQuery:
    def test_query_returns_answer(self, temp_wiki):
        temp_wiki._generate_answer = MagicMock(
            return_value=WikiQueryAnswer(
                answer="The answer is 42.",
                sources_used=["sources/test_doc.md"],
                confidence="high",
                gaps=[],
                suggested_followups=[],
            )
        )
        result = temp_wiki.query("What is the answer?")
        assert result["answer"] == "The answer is 42."
        assert result["confidence"] == "high"

    def test_query_files_answer(self, temp_wiki):
        temp_wiki._generate_answer = MagicMock(
            return_value=WikiQueryAnswer(
                answer="Filed answer.",
                sources_used=[],
                confidence="medium",
                gaps=[],
                suggested_followups=[],
            )
        )
        result = temp_wiki.query("What?", file_answer=True)
        assert "filed_page" in result
        assert result["filed_page"] is not None


class TestLint:
    def test_lint_finds_orphans_and_broken(self, temp_wiki):
        temp_wiki._generate_lint_report = MagicMock(return_value=LintReport())
        temp_wiki._write_page(temp_wiki.entities_dir / "alice.md", "# Alice\n\n[[Bob]]")
        report = temp_wiki.lint(auto_fix=False)
        # alice.md has no inbound links, so it is an orphan
        assert "entities/alice.md" in report["orphan_pages"]
        # Bob is a broken link (no bob.md page exists)
        assert "Bob" in report["broken_links"]

    def test_lint_auto_fix_creates_stubs(self, temp_wiki):
        temp_wiki._generate_lint_report = MagicMock(return_value=LintReport())
        temp_wiki._write_page(temp_wiki.entities_dir / "alice.md", "# Alice\n\n[[Missing]]")
        report = temp_wiki.lint(auto_fix=True)
        assert report["fixes_applied"] >= 1
        stub = temp_wiki.concepts_dir / "missing.md"
        assert stub.exists()
        assert "status: stub" in stub.read_text(encoding="utf-8")


class TestStatus:
    def test_status_counts_pages(self, temp_wiki):
        temp_wiki._write_page(temp_wiki.sources_dir / "s1.md", "# S1")
        temp_wiki._write_page(temp_wiki.entities_dir / "e1.md", "# E1")
        status = temp_wiki.status()
        assert status["total_pages"] == 2
        assert status["page_counts"]["sources"] == 1
        assert status["page_counts"]["entities"] == 1

    def test_status_shows_wiki_path(self, temp_wiki):
        status = temp_wiki.status()
        assert "test_collection" in status["wiki_path"]


class TestPartGrouping:
    def test_doc_base_and_part_recognizes_suffixes(self):
        cases = [
            ("report_part1", "report", 1),
            ("report_part_2", "report", 2),
            ("Report Part 3", "Report", 3),
            ("report_p4", "report", 4),
            ("report (1)", "report", 1),
            ("report (part 2)", "report", 2),
            ("report_vol1", "report", 1),
            ("report_chapter_10", "report", 10),
            ("my_doc_sec5", "my_doc", 5),
        ]
        for name, base, part in cases:
            assert WikiManager._doc_base_and_part(name) == (base, part), name

    def test_doc_base_and_part_leaves_plain_docs_alone(self):
        # Standalone document names must not be treated as parts.
        for name in ("report", "annual_report_2023", "Chapter 1", "setup", "report_v2"):
            base, part = WikiManager._doc_base_and_part(name)
            assert part is None, name
            assert base == name, name

    def test_group_documents_concatenates_parts_in_order(self):
        texts = {
            "report_part2": "SECOND",
            "report_part1": "FIRST",
            "standalone": "ALONE",
        }
        merged = WikiManager._group_documents_by_base(texts)
        # Parts merged under the base name, in ascending part order.
        assert merged["report"] == "FIRST\n\nSECOND"
        # Standalone doc unchanged.
        assert merged["standalone"] == "ALONE"

    def test_group_documents_renames_lone_part_to_base(self):
        # A single part with no siblings is still renamed to its base name.
        merged = WikiManager._group_documents_by_base({"report_part1": "ONLY"})
        assert list(merged.keys()) == ["report"]
        assert merged["report"] == "ONLY"


class TestIngestFromEmbeddingsParts:
    def test_parts_produce_single_page(self, tmp_path):
        """Embedded document parts merge into one wiki source page."""
        import numpy as np
        from libbydbot.brain.embed import DocEmbedder

        with patch(
            "libbydbot.brain.embed.DocEmbedder._generate_embedding"
        ) as mocked, patch(
            "libbydbot.brain.embed.DocEmbedder._get_embedding_dimension",
            return_value=1024,
        ):
            mocked.return_value = np.zeros(1024).tolist()
            db_path = tmp_path / "emb.db"
            embedder = DocEmbedder(
                "test_collection",
                dburl=f"sqlite:///{db_path}",
                embedding_model="mxbai-embed-large",
            )
            embedder.embed_text("Part one intro.", "report_part1", 0)
            embedder.embed_text("Part one body.", "report_part1", 1)
            embedder.embed_text("Part two body.", "report_part2", 0)

        wiki = WikiManager(
            collection_name="test_collection",
            wiki_base=str(tmp_path / "wiki"),
            model="llama3.2",
        )
        seen = []

        def _fake_summary(doc_name, doc_content):
            seen.append((doc_name, doc_content))
            return SourceSummary(
                title=doc_name,
                summary=f"summary of {doc_name}",
                key_takeaways=[],
                entities=[],
                concepts=[],
                contradictions=[],
                questions_raised=[],
            )

        wiki._generate_source_summary = MagicMock(side_effect=_fake_summary)
        wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(
                source_title="x",
                pages_to_update=[],
                pages_to_link=[],
                synthesis_notes="",
            )
        )

        result = wiki.ingest_from_embeddings(embedder, collection="test_collection")

        # One collective page despite three embedded chunks across two parts.
        assert result["documents_ingested"] == 1
        names = [n for n, _ in seen]
        assert names == ["report"]
        # Concatenated content preserves part order (part1 before part2).
        assert "Part one body." in seen[0][1]
        assert "Part two body." in seen[0][1]
        assert seen[0][1].index("Part one intro.") < seen[0][1].index("Part two body.")
        assert (wiki.sources_dir / "report.md").exists()
        # No per-part page should have been created.
        assert not (wiki.sources_dir / "report_part1.md").exists()
        assert not (wiki.sources_dir / "report_part2.md").exists()

    def test_merge_parts_disabled_keeps_separate(self, tmp_path):
        """With merge_parts=False, parts stay as separate pages."""
        import numpy as np
        from libbydbot.brain.embed import DocEmbedder

        with patch(
            "libbydbot.brain.embed.DocEmbedder._generate_embedding"
        ) as mocked, patch(
            "libbydbot.brain.embed.DocEmbedder._get_embedding_dimension",
            return_value=1024,
        ):
            mocked.return_value = np.zeros(1024).tolist()
            db_path = tmp_path / "emb.db"
            embedder = DocEmbedder(
                "test_collection",
                dburl=f"sqlite:///{db_path}",
                embedding_model="mxbai-embed-large",
            )
            embedder.embed_text("Part one.", "report_part1", 0)
            embedder.embed_text("Part two.", "report_part2", 0)

        wiki = WikiManager(
            collection_name="test_collection",
            wiki_base=str(tmp_path / "wiki"),
            model="llama3.2",
        )
        wiki._generate_source_summary = MagicMock(
            return_value=SourceSummary(
                title="x",
                summary="s",
                key_takeaways=[],
                entities=[],
                concepts=[],
                contradictions=[],
                questions_raised=[],
            )
        )
        wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(
                source_title="x",
                pages_to_update=[],
                pages_to_link=[],
                synthesis_notes="",
            )
        )

        result = wiki.ingest_from_embeddings(
            embedder, collection="test_collection", merge_parts=False
        )
        assert result["documents_ingested"] == 2


class TestConsolidate:
    def test_consolidate_merges_part_pages_and_rewrites_links(self, temp_wiki):
        # Two per-part source pages for the same base document.
        temp_wiki._write_page(
            temp_wiki.sources_dir / "report_part1.md",
            "---\ntitle: Report Part 1\n---\n\n# Report Part 1\n\n"
            "First section content.\n\n## Summary\n\nPart one summary.\n",
        )
        temp_wiki._write_page(
            temp_wiki.sources_dir / "report_part2.md",
            "---\ntitle: Report Part 2\n---\n\n# Report Part 2\n\n"
            "Second section content.\n",
        )
        # A standalone source page that must be left untouched.
        temp_wiki._write_page(
            temp_wiki.sources_dir / "other.md", "# Other\n\nUnrelated doc.\n"
        )
        # An entity page linking to a part page.
        temp_wiki._write_page(
            temp_wiki.entities_dir / "topic.md",
            "# Topic\n\nSee [[report_part1]] and [[report_part2|the report]].\n",
        )

        result = temp_wiki.consolidate_part_pages()

        assert result["groups_merged"] == 1
        assert result["pages_removed"] == 2
        # Merged collective page exists under the base name.
        merged = temp_wiki.sources_dir / "report.md"
        assert merged.exists()
        merged_text = merged.read_text(encoding="utf-8")
        assert "Part one summary." in merged_text
        assert "Second section content." in merged_text
        assert "merged_from" in merged_text
        # Per-part pages are gone.
        assert not (temp_wiki.sources_dir / "report_part1.md").exists()
        assert not (temp_wiki.sources_dir / "report_part2.md").exists()
        # Standalone page is untouched.
        assert (temp_wiki.sources_dir / "other.md").exists()
        # Links were rewritten to the merged page.
        topic = temp_wiki.entities_dir / "topic.md"
        topic_text = topic.read_text(encoding="utf-8")
        assert "[[report]]" in topic_text
        assert "[[report|the report]]" in topic_text
        assert "report_part1" not in topic_text

    def test_consolidate_noop_without_part_pages(self, temp_wiki):
        temp_wiki._write_page(
            temp_wiki.sources_dir / "plain.md", "# Plain\n\nNo parts here.\n"
        )
        result = temp_wiki.consolidate_part_pages()
        assert result["groups_merged"] == 0
        assert result["pages_removed"] == 0
        assert (temp_wiki.sources_dir / "plain.md").exists()
