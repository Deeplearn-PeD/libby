import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libbydbot.brain.graph import WikiKnowledgeGraph
from libbydbot.brain.wiki import WikiManager
from libbydbot.brain.wiki_models import (
    KeyConcept,
    KeyEntity,
    SourceSummary,
    WikiQueryAnswer,
    WikiUpdatePlan,
)


@pytest.fixture
def wiki_dir():
    """Create a temporary wiki directory with a small set of pages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wiki = Path(tmpdir)
        for sub in ("sources", "entities", "concepts", "synthesis"):
            (wiki / sub).mkdir(parents=True)
        (wiki / "sources" / "doc_a.md").write_text(
            "---\ntitle: Doc A\ndate_ingested: 2026-01-01\n---\n\n"
            "# Doc A\n\nAbout [[Alice]] and [[Testing]].\n",
            encoding="utf-8",
        )
        (wiki / "entities" / "alice.md").write_text(
            "---\ntitle: Alice\nentity_type: person\n---\n\n"
            "# Alice\n\nFriend of [[Bob]]. Mentioned in [[doc_a|Doc A]].\n",
            encoding="utf-8",
        )
        (wiki / "entities" / "bob.md").write_text(
            "---\ntitle: Bob\n---\n\n# Bob\n", encoding="utf-8"
        )
        (wiki / "concepts" / "testing.md").write_text(
            "---\ntitle: Testing\n---\n\n# Testing\n\nRelated: [[Code Review]].\n",
            encoding="utf-8",
        )
        (wiki / "index.md").write_text("# Index\n", encoding="utf-8")
        (wiki / "log.md").write_text("# Log\n", encoding="utf-8")
        yield wiki


@pytest.fixture
def kg(wiki_dir):
    return WikiKnowledgeGraph(wiki_dir, "test_collection").rebuild()


def make_summary(title="Doc B"):
    return SourceSummary(
        title=title,
        summary="s",
        key_takeaways=[],
        entities=[
            KeyEntity(
                name="Carol",
                entity_type="person",
                description="d",
                related_entities=["Alice"],
            )
        ],
        concepts=[
            KeyConcept(name="Testing", description="t", related_concepts=["QA"])
        ],
        contradictions=[],
        questions_raised=[],
    )


CHUNKS = [
    {"doc_hash": "h1", "doc_name": "Doc B", "page_number": 0,
     "content": "Carol talks about Testing here."},
    {"doc_hash": "h2", "doc_name": "Doc B", "page_number": 1,
     "content": "Nothing relevant."},
]


class TestRebuild:
    def test_page_nodes_created(self, kg):
        assert "sources/doc_a.md" in kg.graph
        assert "entities/alice.md" in kg.graph
        assert "entities/bob.md" in kg.graph
        assert "concepts/testing.md" in kg.graph

    def test_index_and_log_excluded(self, kg):
        assert not any("index" in n for n in kg.graph.nodes)
        assert not any("log" in n for n in kg.graph.nodes)

    def test_node_attributes(self, kg):
        data = kg.graph.nodes["entities/alice.md"]
        assert data["title"] == "Alice"
        assert data["node_type"] == "entity"

    def test_wikilink_edges(self, kg):
        assert kg.graph.has_edge("sources/doc_a.md", "entities/alice.md")
        assert (
            kg.graph.edges["sources/doc_a.md", "entities/alice.md"]["edge_type"]
            == "mentions"
        )

    def test_alias_link_resolved(self, kg):
        # [[doc_a|Doc A]] should resolve to the source page, not a stub
        assert kg.graph.has_edge("entities/alice.md", "sources/doc_a.md")

    def test_broken_link_becomes_stub(self, kg):
        assert "stub:code_review" in kg.graph
        assert kg.graph.nodes["stub:code_review"]["node_type"] == "stub"

    def test_stub_edge_inherits_source_semantics(self, kg):
        # concept -> stub should be related_to, not plain links_to
        assert (
            kg.graph.edges["concepts/testing.md", "stub:code_review"]["edge_type"]
            == "related_to"
        )

    def test_entity_to_source_edge_type(self, kg):
        assert (
            kg.graph.edges["entities/alice.md", "sources/doc_a.md"]["edge_type"]
            == "mentioned_in"
        )

    def test_entity_to_entity_related(self, kg):
        assert (
            kg.graph.edges["entities/alice.md", "entities/bob.md"]["edge_type"]
            == "related_to"
        )


class TestPersistence:
    def test_save_and_load(self, wiki_dir):
        kg = WikiKnowledgeGraph(wiki_dir, "c").rebuild()
        kg2 = WikiKnowledgeGraph(wiki_dir, "c")
        assert kg2.graph.number_of_nodes() == kg.graph.number_of_nodes()
        assert kg2.graph.number_of_edges() == kg.graph.number_of_edges()

    def test_rebuild_preserves_chunks(self, wiki_dir):
        kg = WikiKnowledgeGraph(wiki_dir, "c").rebuild()
        kg.update_from_ingest("Doc B", make_summary(), CHUNKS)
        # Simulate WikiManager writing the pages after ingest
        (wiki_dir / "sources" / "doc_b.md").write_text(
            "---\ntitle: Doc B\n---\n\n# Doc B\n\n[[Carol]] [[Testing]]\n",
            encoding="utf-8",
        )
        (wiki_dir / "entities" / "carol.md").write_text(
            "---\ntitle: Carol\n---\n\n# Carol\n", encoding="utf-8"
        )
        kg.rebuild()
        assert "chunk:h1" in kg.graph
        assert kg.graph.has_edge("chunk:h1", kg.resolve_node("Carol"))
        assert kg.graph.has_edge("chunk:h1", kg.resolve_node("Testing"))


class TestIngestUpdate:
    def test_update_creates_nodes_and_edges(self, wiki_dir):
        kg = WikiKnowledgeGraph(wiki_dir, "c").rebuild()
        kg.update_from_ingest("Doc B", make_summary(), CHUNKS)
        assert "sources/doc_b.md" in kg.graph
        carol = kg.resolve_node("Carol")
        assert carol is not None
        assert kg.graph.has_edge("sources/doc_b.md", carol)
        # Related entities/concepts produce stubs and edges
        assert kg.resolve_node("QA") is not None

    def test_chunks_linked_by_name_match(self, wiki_dir):
        kg = WikiKnowledgeGraph(wiki_dir, "c").rebuild()
        kg.update_from_ingest("Doc B", make_summary(), CHUNKS)
        assert kg.graph.has_edge("chunk:h1", kg.resolve_node("Carol"))
        assert kg.graph.has_edge("chunk:h1", kg.resolve_node("Testing"))
        assert not kg.graph.has_edge("chunk:h2", kg.resolve_node("Carol"))

    def test_chunk_of_edge_to_source(self, wiki_dir):
        kg = WikiKnowledgeGraph(wiki_dir, "c").rebuild()
        kg.update_from_ingest("Doc B", make_summary(), CHUNKS)
        assert kg.graph.has_edge("chunk:h1", "sources/doc_b.md")


class TestQueries:
    def test_shortest_path_found(self, kg):
        result = kg.shortest_path("Alice", "Testing")
        assert result["found"]
        assert result["length"] >= 1
        assert result["nodes"][0]["title"] == "Alice"
        assert result["nodes"][-1]["title"] == "Testing"

    def test_shortest_path_missing_node(self, kg):
        result = kg.shortest_path("Alice", "NoSuchThing")
        assert not result["found"]
        assert "NoSuchThing" in result["error"]

    def test_explain(self, kg):
        result = kg.explain("Alice")
        assert result["found"]
        assert result["node_type"] == "entity"
        assert result["degree"] > 0
        assert result["outbound"] or result["inbound"]

    def test_explain_missing(self, kg):
        result = kg.explain("Nobody")
        assert not result["found"]

    def test_score_pages_ranks_relevant(self, kg):
        ranked = kg.score_pages("What does Alice do?")
        assert ranked
        assert "entities/alice.md" in ranked

    def test_score_pages_empty_question(self, kg):
        assert kg.score_pages("") == []

    def test_subgraph_for_query(self, kg):
        sub = kg.subgraph_for_query("Alice testing")
        assert sub["nodes"]
        assert isinstance(sub["edges"], list)

    def test_hubs(self, kg):
        hubs = kg.hubs(3)
        assert hubs
        assert all("degree" in h for h in hubs)
        assert hubs[0]["degree"] >= hubs[-1]["degree"]

    def test_chunks_for_node(self, wiki_dir):
        kg = WikiKnowledgeGraph(wiki_dir, "c").rebuild()
        kg.update_from_ingest("Doc B", make_summary(), CHUNKS)
        chunks = kg.chunks_for_node("Carol")
        assert chunks
        assert chunks[0]["doc_hash"] == "h1"

    def test_status(self, kg):
        status = kg.status()
        assert status["total_nodes"] == kg.graph.number_of_nodes()
        assert status["total_edges"] == kg.graph.number_of_edges()
        assert "entity" in status["node_counts"]
        assert "hubs" in status


class TestExport:
    def test_export_html(self, kg):
        path = kg.export_html()
        assert path.exists()
        assert path.stat().st_size > 1000
        content = path.read_text(encoding="utf-8")
        assert "<html" in content.lower()

    def test_export_html_custom_path(self, kg, tmp_path):
        out = tmp_path / "custom.html"
        path = kg.export_html(out)
        assert path == out
        assert out.exists()


class TestWikiIntegration:
    @pytest.fixture
    def graph_wiki(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = WikiManager(
                collection_name="graph_coll",
                wiki_base=tmpdir,
                model="llama3.2",
                graph_enabled=True,
            )
            wiki._llm = MagicMock()
            yield wiki

    def _mock_ingest(self, wiki):
        wiki._generate_source_summary = MagicMock(return_value=make_summary())
        wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(
                source_title="Doc B",
                pages_to_update=[],
                pages_to_link=[],
                synthesis_notes="",
            )
        )

    def test_ingest_builds_graph(self, graph_wiki):
        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            result = graph_wiki.ingest_source("Doc B", "Carol talks about Testing here.")

        assert result["source"] == "Doc B"
        kg = graph_wiki._get_knowledge_graph()
        assert "sources/doc_b.md" in kg.graph
        assert kg.resolve_node("Carol") is not None
        assert "chunk:h1" in kg.graph
        assert (graph_wiki.wiki_dir / "graph.json").exists()

    def test_fetch_chunks_respects_embed_db_env(self, graph_wiki, monkeypatch):
        monkeypatch.setenv("EMBED_DB", "sqlite:////tmp/env_embed.db")
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            chunks = graph_wiki._fetch_document_chunks("Doc B", "some content")
        assert chunks == CHUNKS
        assert mock_de.call_args.kwargs["dburl"] == "sqlite:////tmp/env_embed.db"

    def test_ingest_without_embedded_chunks_falls_back(self, graph_wiki):
        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = []
            graph_wiki.ingest_source("Doc B", "Carol talks about Testing. " * 50)

        kg = graph_wiki._get_knowledge_graph()
        # Fallback chunking should still create chunk nodes
        chunk_nodes = [n for n in kg.graph.nodes if n.startswith("chunk:")]
        assert chunk_nodes

    def test_graph_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki = WikiManager(
                collection_name="nogrph",
                wiki_base=tmpdir,
                model="llama3.2",
                graph_enabled=False,
            )
            wiki._llm = MagicMock()
            self._mock_ingest(wiki)
            wiki.ingest_source("Doc B", "Some content.")
            assert not (wiki.wiki_dir / "graph.json").exists()

    def test_ingest_captures_page_body_links(self, graph_wiki):
        # First ingest creates the Alice entity page
        graph_wiki._generate_source_summary = MagicMock(
            return_value=SourceSummary(
                title="Doc 1", summary="s", key_takeaways=[],
                entities=[KeyEntity(name="Alice", entity_type="person",
                                    description="d", related_entities=[])],
                concepts=[], contradictions=[], questions_raised=[],
            )
        )
        graph_wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(source_title="Doc 1", pages_to_update=[],
                                        pages_to_link=[], synthesis_notes="")
        )
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = []
            graph_wiki.ingest_source("Doc 1", "Alice content.")

        # Second ingest: Bob is related to Alice (written as wikilink in Bob's page)
        graph_wiki._generate_source_summary = MagicMock(
            return_value=SourceSummary(
                title="Doc 2", summary="s", key_takeaways=[],
                entities=[KeyEntity(name="Bob", entity_type="person",
                                    description="d", related_entities=["Alice"])],
                concepts=[], contradictions=[], questions_raised=[],
            )
        )
        graph_wiki._generate_update_plan = MagicMock(
            return_value=WikiUpdatePlan(source_title="Doc 2", pages_to_update=[],
                                        pages_to_link=[], synthesis_notes="")
        )
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = []
            graph_wiki.ingest_source("Doc 2", "Bob content.")

        kg = graph_wiki._get_knowledge_graph()
        assert kg.graph.has_edge("entities/bob.md", "entities/alice.md")
        assert (
            kg.graph.edges["entities/bob.md", "entities/alice.md"]["edge_type"]
            == "related_to"
        )

    def test_graph_rebuild_and_status(self, graph_wiki):
        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            graph_wiki.ingest_source("Doc B", "Carol talks about Testing here.")

        status = graph_wiki.graph_rebuild()
        assert status["total_nodes"] > 0
        status = graph_wiki.graph_status()
        assert status["collection"] == "graph_coll"

    def test_graph_path_and_explain(self, graph_wiki):
        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            graph_wiki.ingest_source("Doc B", "Carol talks about Testing here.")

        result = graph_wiki.graph_path("Carol", "Testing")
        assert result["found"]
        result = graph_wiki.graph_explain("Carol")
        assert result["found"]
        assert result["node_type"] == "entity"

    def test_graph_query_and_export(self, graph_wiki):
        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            graph_wiki.ingest_source("Doc B", "Carol talks about Testing here.")

        sub = graph_wiki.graph_query("What does Carol know about Testing?")
        assert sub["nodes"]
        html = graph_wiki.graph_export_html()
        assert html.exists()

    def test_query_uses_graph_ranking(self, graph_wiki):
        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            graph_wiki.ingest_source("Doc B", "Carol talks about Testing here.")

        graph_wiki._generate_answer = MagicMock(
            return_value=WikiQueryAnswer(
                answer="Answer.",
                sources_used=["entities/carol.md"],
                confidence="high",
                gaps=[],
                suggested_followups=[],
            )
        )
        result = graph_wiki.query("What does Carol know about Testing?")
        assert result["answer"] == "Answer."

    def test_status_includes_graph(self, graph_wiki):
        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            graph_wiki.ingest_source("Doc B", "Carol talks about Testing here.")

        status = graph_wiki.status()
        assert status["graph"] is not None
        assert status["graph"]["total_nodes"] > 0

    def test_lint_includes_graph_hubs(self, graph_wiki):
        from libbydbot.brain.wiki_models import LintReport

        self._mock_ingest(graph_wiki)
        with patch("libbydbot.brain.embed.DocEmbedder") as mock_de:
            mock_de.return_value.get_document_chunks.return_value = CHUNKS
            graph_wiki.ingest_source("Doc B", "Carol talks about Testing here.")

        graph_wiki._generate_lint_report = MagicMock(return_value=LintReport())
        report = graph_wiki.lint(auto_fix=False)
        assert "graph_hubs" in report
