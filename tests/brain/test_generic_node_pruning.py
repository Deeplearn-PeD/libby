"""
Tests for generic-node pruning in the wiki knowledge graph.

Junk names (session identifiers, single letters, pure numbers) must not
become graph nodes — including legacy wiki pages written before the
name-quality filter existed — while session SOURCE pages stay as
traceability anchors and legitimate concepts survive.

Run with: uv run pytest tests/brain/test_generic_node_pruning.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from libbydbot.brain.graph import WikiKnowledgeGraph
from libbydbot.brain.wiki_models import (
    KeyConcept,
    KeyEntity,
    SourceSummary,
)

def _write_page(wiki: Path, sub: str, name: str, content: str) -> None:
    page = wiki / sub / f"{name}.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(content, encoding="utf-8")


def _make_wiki(tmp_path: Path) -> Path:
    """Wiki with legitimate pages AND legacy junk pages."""
    wiki = tmp_path / "wiki"
    for sub in ("sources", "entities", "concepts", "synthesis"):
        (wiki / sub).mkdir(parents=True)

    # Session source page (structural anchor, kept)
    _write_page(
        wiki, "sources", "session_123",
        "---\ntitle: session-123\ndate_ingested: 2026-01-01\nsource_type: chat\n---\n\n"
        "# session-123\n\nDiscussion of [[Dengue]] and [[Vaccine]].\n",
    )
    # Legitimate entity + concept
    _write_page(
        wiki, "entities", "vaccine",
        "---\ntitle: Vaccine\nentity_type: intervention\n---\n\n# Vaccine\n\n"
        "Discussed in [[session_123|session-123]].\n",
    )
    _write_page(
        wiki, "concepts", "dengue",
        "---\ntitle: Dengue\n---\n\n# Dengue\n\nRelated: [[Vaccine]].\n",
    )
    # Legacy junk entity pages (pre-filter extraction noise)
    _write_page(
        wiki, "entities", "session_456",
        "---\ntitle: session-456\nentity_type: session\n---\n\n# session-456\n",
    )
    _write_page(wiki, "entities", "a", "---\ntitle: A\n---\n\n# A\n")
    _write_page(wiki, "entities", "350045", "---\ntitle: 350045\n---\n\n# 350045\n")
    # Junk concept page
    _write_page(wiki, "concepts", "12", "---\ntitle: 12\n---\n\n# 12\n")
    # Legitimate concept that ONLY linked to a junk entity → becomes
    # isolated after pruning and is removed too.
    _write_page(
        wiki, "concepts", "orphan",
        "---\ntitle: Orphan\n---\n\n# Orphan\n\nSee [[session_456|session-456]].\n",
    )
    _write_page(wiki, "index.md", "__dummy__", "# Index\n")
    return wiki


class TestGenericNodePruning:
    """Graph build drops generic-named nodes; anchors and concepts survive."""

    @pytest.fixture
    def kg(self, tmp_path):
        wiki = _make_wiki(tmp_path)
        return WikiKnowledgeGraph(wiki, "test_collection").rebuild()

    def _node_types(self, kg):
        return {n: d.get("node_type", "") for n, d in kg.graph.nodes(data=True)}

    def test_generic_entity_nodes_removed(self, kg):
        types = self._node_types(kg)
        assert "entities/session_456.md" not in types
        assert "entities/a.md" not in types
        assert "entities/350045.md" not in types

    def test_generic_concept_nodes_removed(self, kg):
        types = self._node_types(kg)
        assert "concepts/12.md" not in types

    def test_session_source_page_kept_as_anchor(self, kg):
        types = self._node_types(kg)
        assert types.get("sources/session_123.md") == "source"

    def test_legitimate_pages_survive(self, kg):
        types = self._node_types(kg)
        assert "entities/vaccine.md" in types
        assert "concepts/dengue.md" in types
        assert types.get("concepts/dengue.md") == "concept"

    def test_isolated_orphan_concept_pruned(self, kg):
        # 'Orphan' only linked to the removed junk entity session-456.
        types = self._node_types(kg)
        assert "concepts/orphan.md" not in types

    def test_legitimate_edges_preserved(self, kg):
        # Dengue concept → Vaccine entity (related) survives pruning.
        assert kg.graph.has_edge(
            "concepts/dengue.md", "entities/vaccine.md"
        ) or kg.graph.has_edge("entities/vaccine.md", "concepts/dengue.md")


class TestIngestSkipsGenericNames:
    """ingest_source drops generic entity/concept names before writing pages."""

    def _wiki_manager(self, tmp_path):
        from libbydbot.brain.wiki import WikiManager

        return WikiManager(str(tmp_path / "wiki"), "test_collection")

    def test_generic_names_never_become_pages(self, tmp_path, monkeypatch):
        summary = SourceSummary(
            title="session-9",
            summary="s",
            key_takeaways=[],
            entities=[
                KeyEntity(name="Oswaldo Cruz", entity_type="person", description="x"),
                KeyEntity(name="A", entity_type="concept", description="junk"),
                KeyEntity(name="123", entity_type="identifier", description="junk"),
            ],
            concepts=[
                KeyConcept(name="Vector control", description="x"),
                KeyConcept(name="session-77", description="junk"),
            ],
            contradictions=[],
            questions_raised=[],
        )
        manager = self._wiki_manager(tmp_path)
        monkeypatch.setattr(
            manager, "_generate_source_summary", lambda doc_name, doc_content: summary
        )
        monkeypatch.setattr(
            manager, "_generate_update_plan",
            lambda doc_name, summary: MagicMock(),
        )
        monkeypatch.setattr(
            manager, "_update_knowledge_graph", lambda *a, **k: None
        )

        result = manager.ingest_source("session-9", "user: talk about dengue")

        # junk filtered, legit kept
        assert result["skipped_generic"] == 3
        assert result["entities_created"] == 1
        assert result["concepts_created"] == 1
        assert not (manager.entities_dir / "a.md").exists()
        assert not (manager.entities_dir / "123.md").exists()
        assert not (manager.concepts_dir / "session_77.md").exists()
        assert (manager.entities_dir / "oswaldo_cruz.md").exists()
        assert (manager.concepts_dir / "vector_control.md").exists()
