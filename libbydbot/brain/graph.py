"""
Knowledge graph layer for Libby's LLM Wiki.

WikiKnowledgeGraph builds and maintains a persistent NetworkX graph over a
collection's wiki pages (sources, entities, concepts, synthesis) and its
embedded chunks. Nodes carry page metadata from YAML frontmatter; edges
capture wikilinks, mention/related relationships, and chunk references.

The graph is persisted as ``graph.json`` (NetworkX node-link format) inside
the wiki directory and can be exported to an interactive ``graph.html``
visualization via pyvis.
"""

import itertools
import json
import re
from pathlib import Path
from typing import Any

import loguru
import networkx as nx

logger = loguru.logger

PAGE_NODE_TYPES = ("source", "entity", "concept", "synthesis")

DIR_TO_NODE_TYPE = {
    "sources": "source",
    "entities": "entity",
    "concepts": "concept",
    "synthesis": "synthesis",
}

NODE_COLORS = {
    "source": "#4C9AFF",
    "entity": "#36B37E",
    "concept": "#FFAB00",
    "synthesis": "#FF5630",
    "stub": "#9AA1B0",
    "chunk": "#C0B6F2",
}

EDGE_COLORS = {
    "mentions": "#4C9AFF",
    "mentioned_in": "#57D9A3",
    "related_to": "#36B37E",
    "links_to": "#9AA1B0",
    "chunk_of": "#D0D5DD",
    "references": "#B497E7",
}


def sanitize_name(name: str) -> str:
    """Convert a name to a filesystem-safe page name (matches WikiManager)."""
    return re.sub(r"[^\w\-]", "_", name).lower()


class WikiKnowledgeGraph:
    """
    Knowledge graph over a single collection's wiki and embedded chunks.

    Page nodes are identified by their path relative to the wiki directory
    (e.g. ``entities/alice.md``); chunk nodes by ``chunk:<doc_hash>``.
    """

    def __init__(self, wiki_dir: str | Path, collection_name: str = ""):
        self.wiki_dir = Path(wiki_dir)
        self.collection_name = collection_name
        self.graph = nx.DiGraph()
        if self.graph_path.exists():
            try:
                self.load()
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load graph.json ({e}), starting fresh")
                self.graph = nx.DiGraph()

    # ────────────────────────── properties ──────────────────────────

    @property
    def graph_path(self) -> Path:
        return self.wiki_dir / "graph.json"

    # ────────────────────────── persistence ─────────────────────────

    def save(self) -> Path:
        """Persist the graph to graph.json in node-link format."""
        data = nx.node_link_data(self.graph, edges="edges")
        self.graph_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        logger.info(f"Saved knowledge graph: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges -> {self.graph_path}")
        return self.graph_path

    def load(self) -> None:
        """Load the graph from graph.json."""
        data = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.graph = nx.node_link_graph(data, edges="edges")

    # ────────────────────────── parsing helpers ─────────────────────

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
        """Extract YAML frontmatter and body from markdown content."""
        import yaml

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    fm = {}
                return fm, parts[2].strip()
        return {}, content.strip()

    @staticmethod
    def _extract_wikilinks(content: str) -> list[str]:
        """Extract wikilink targets (without aliases) from markdown content."""
        links = re.findall(r"\[\[([^\]]+)\]\]", content)
        return [link.split("|")[0].strip() for link in links]

    def _node_type_for_path(self, rel_path: str) -> str | None:
        top_dir = rel_path.split("/")[0]
        return DIR_TO_NODE_TYPE.get(top_dir)

    def _edge_type(self, u_type: str, v_type: str) -> str:
        """Derive a relationship type from the connected node types."""
        if u_type == "chunk" and v_type == "source":
            return "chunk_of"
        if u_type == "chunk":
            return "references"
        # Stubs stand in for a not-yet-created page; assume the source's type
        # so relationship semantics (mentions/related_to) are preserved.
        if v_type == "stub":
            v_type = u_type
        if u_type == "source" and v_type in ("entity", "concept"):
            return "mentions"
        if u_type in ("entity", "concept") and v_type == "source":
            return "mentioned_in"
        if u_type == v_type and u_type in ("entity", "concept"):
            return "related_to"
        return "links_to"

    def resolve_node(self, name: str) -> str | None:
        """Resolve a human name or page title to a node id (exact, case-insensitive)."""
        name_l = name.strip().lower()
        if name_l in self.graph:
            return name_l
        for node, data in self.graph.nodes(data=True):
            if data.get("title", "").lower() == name_l:
                return node
            if Path(node).stem.lower() == name_l:
                return node
        return None

    def _resolve_fuzzy(self, name: str) -> str | None:
        """Resolve a name to a node id, falling back to partial title matching."""
        resolved = self.resolve_node(name)
        if resolved is not None:
            return resolved
        name_l = name.strip().lower()
        for node, data in self.graph.nodes(data=True):
            title = data.get("title", "").lower()
            if title and (name_l in title or title in name_l):
                return node
        return None

    # ────────────────────────── building ────────────────────────────

    def _add_page_node(self, rel_path: str, content: str) -> str:
        """Add (or update) a node for a wiki page. Returns the node id."""
        fm, _ = self._parse_frontmatter(content)
        node_type = self._node_type_for_path(rel_path) or "synthesis"
        title = fm.get("title", Path(rel_path).stem)
        self.graph.add_node(
            rel_path,
            title=title,
            node_type=node_type,
            path=str(self.wiki_dir / rel_path),
            date=str(fm.get("date_created", fm.get("date_ingested", ""))),
        )
        return rel_path

    def _link_target_node(self, target: str) -> str:
        """Resolve a wikilink target to a node id, creating a stub if missing."""
        resolved = self.resolve_node(target)
        if resolved is not None:
            return resolved
        stub_id = f"stub:{sanitize_name(target)}"
        if stub_id not in self.graph:
            self.graph.add_node(stub_id, title=target, node_type="stub", path="", date="")
        return stub_id

    def rebuild(self) -> "WikiKnowledgeGraph":
        """
        Rebuild the page-level graph from the wiki directory on disk.

        Chunk nodes and their edges are preserved across rebuilds.
        """
        chunk_nodes = {
            n: dict(d)
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "chunk"
        }
        old_titles = {n: d.get("title", "") for n, d in self.graph.nodes(data=True)}
        chunk_refs = [
            (u, v, dict(d))
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "references" and u in chunk_nodes
        ]

        self.graph = nx.DiGraph()

        pages: dict[str, str] = {}
        for md_file in self.wiki_dir.rglob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            rel = md_file.relative_to(self.wiki_dir).as_posix()
            if self._node_type_for_path(rel) is None:
                continue
            try:
                pages[rel] = md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Skipping unreadable wiki page {md_file}: {e}")

        for rel, content in pages.items():
            self._add_page_node(rel, content)

        for rel, content in pages.items():
            u_type = self.graph.nodes[rel].get("node_type", "")
            for target in self._extract_wikilinks(content):
                if not target:
                    continue
                v = self._link_target_node(target)
                if v == rel:
                    continue
                v_type = self.graph.nodes[v].get("node_type", "")
                self.graph.add_edge(rel, v, edge_type=self._edge_type(u_type, v_type))

        # Restore chunk nodes and re-attach them to their source pages.
        # Reference targets may have changed identity (e.g. stub -> page),
        # so re-resolve them by title when the original id is gone.
        for node, attrs in chunk_nodes.items():
            self.graph.add_node(node, **attrs)
        for u, v, attrs in chunk_refs:
            target = v if v in self.graph else self.resolve_node(old_titles.get(v, ""))
            if u in self.graph and target is not None:
                self.graph.add_edge(u, target, **attrs)
        self._reattach_chunk_of_edges()

        self.save()
        return self

    def _source_node_for_doc(self, doc_name: str) -> str | None:
        candidate = f"sources/{sanitize_name(doc_name)}.md"
        if candidate in self.graph:
            return candidate
        # Fall back to matching by title
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "source" and data.get("title", "").lower() == doc_name.lower():
                return node
        return None

    def _reattach_chunk_of_edges(self) -> None:
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") != "chunk":
                continue
            source_node = self._source_node_for_doc(data.get("doc_name", ""))
            if source_node:
                self.graph.add_edge(node, source_node, edge_type="chunk_of")

    def _ensure_page_node(self, name: str, node_type: str) -> str:
        """
        Get or create a node using the canonical page path for a name.

        Wiki pages are written to deterministic paths by WikiManager, so
        graph nodes should use those ids rather than stub ids.
        """
        resolved = self.resolve_node(name)
        if resolved is not None:
            current_type = self.graph.nodes[resolved].get("node_type")
            if current_type in (node_type, "stub"):
                self.graph.nodes[resolved].update(node_type=node_type)
                if not self.graph.nodes[resolved].get("title"):
                    self.graph.nodes[resolved]["title"] = name
                return resolved
        subdir = "entities" if node_type == "entity" else "concepts"
        node_id = f"{subdir}/{sanitize_name(name)}.md"
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                title=name,
                node_type=node_type,
                path=str(self.wiki_dir / node_id),
                date="",
            )
        else:
            self.graph.nodes[node_id].update(node_type=node_type)
        return node_id

    def update_from_ingest(
        self,
        doc_name: str,
        summary,
        chunks: list[dict] | None = None,
    ) -> "WikiKnowledgeGraph":
        """
        Incrementally update the graph after a wiki ingest.

        :param doc_name: name of the ingested source document
        :param summary: SourceSummary produced during ingest
        :param chunks: optional list of chunk dicts (doc_hash, doc_name,
                       page_number, content) to link to entities/concepts
        """
        source_id = f"sources/{sanitize_name(doc_name)}.md"
        if source_id not in self.graph:
            self.graph.add_node(
                source_id,
                title=summary.title or doc_name,
                node_type="source",
                path=str(self.wiki_dir / source_id),
                date="",
            )

        entity_names: list[str] = []
        for entity in summary.entities:
            node_id = self._ensure_page_node(entity.name, "entity")
            entity_names.append(entity.name)
            self.graph.add_edge(source_id, node_id, edge_type="mentions")
            for related in entity.related_entities:
                rel_id = self._link_target_node(related)
                self.graph.add_edge(
                    node_id, rel_id, edge_type=self._edge_type("entity", self.graph.nodes[rel_id].get("node_type", ""))
                )

        concept_names: list[str] = []
        for concept in summary.concepts:
            node_id = self._ensure_page_node(concept.name, "concept")
            concept_names.append(concept.name)
            self.graph.add_edge(source_id, node_id, edge_type="mentions")
            for related in concept.related_concepts:
                rel_id = self._link_target_node(related)
                self.graph.add_edge(
                    node_id, rel_id, edge_type=self._edge_type("concept", self.graph.nodes[rel_id].get("node_type", ""))
                )

        if chunks:
            self.add_chunks(chunks)
            self.link_chunks_to_names(chunks, entity_names + concept_names)

        self.save()
        return self

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add chunk nodes and link them to their source document node."""
        for chunk in chunks:
            doc_hash = chunk.get("doc_hash", "")
            if not doc_hash:
                continue
            node_id = f"chunk:{doc_hash}"
            if node_id not in self.graph:
                self.graph.add_node(
                    node_id,
                    title=f"{chunk.get('doc_name', 'doc')} #{chunk.get('page_number', 0)}",
                    node_type="chunk",
                    path="",
                    date="",
                    doc_name=chunk.get("doc_name", ""),
                    page_number=chunk.get("page_number", 0),
                    doc_hash=doc_hash,
                )
            source_node = self._source_node_for_doc(chunk.get("doc_name", ""))
            if source_node:
                self.graph.add_edge(node_id, source_node, edge_type="chunk_of")

    def link_chunks_to_names(self, chunks: list[dict], names: list[str]) -> int:
        """
        Create ``references`` edges from chunks to entity/concept nodes
        whose names appear in the chunk text. Returns edges created.
        """
        created = 0
        for name in names:
            target = self.resolve_node(name)
            if target is None or self.graph.nodes[target].get("node_type") not in (
                "entity",
                "concept",
            ):
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
            for chunk in chunks:
                doc_hash = chunk.get("doc_hash", "")
                content = chunk.get("content", "")
                if not doc_hash or not content:
                    continue
                if pattern.search(content):
                    chunk_id = f"chunk:{doc_hash}"
                    if chunk_id in self.graph and not self.graph.has_edge(chunk_id, target):
                        self.graph.add_edge(chunk_id, target, edge_type="references")
                        created += 1
        return created

    # ────────────────────────── queries ─────────────────────────────

    def shortest_path(self, a: str, b: str) -> dict[str, Any]:
        """Find the shortest undirected path between two named nodes."""
        u = self._resolve_fuzzy(a)
        v = self._resolve_fuzzy(b)
        if u is None or v is None:
            missing = a if u is None else b
            return {"found": False, "error": f"Node not found: {missing}", "nodes": [], "edges": []}
        try:
            path = nx.shortest_path(self.graph.to_undirected(), u, v)
        except nx.NetworkXNoPath:
            return {"found": False, "error": f"No path between '{a}' and '{b}'", "nodes": [], "edges": []}
        nodes = [{"id": n, "title": self.graph.nodes[n].get("title", n)} for n in path]
        edges = []
        for x, y in itertools.pairwise(path):
            if self.graph.has_edge(x, y):
                edges.append({"from": x, "to": y, "edge_type": self.graph.edges[x, y].get("edge_type", "links_to")})
            elif self.graph.has_edge(y, x):
                edges.append({"from": y, "to": x, "edge_type": self.graph.edges[y, x].get("edge_type", "links_to")})
        return {"found": True, "length": len(path) - 1, "nodes": nodes, "edges": edges}

    def explain(self, name: str) -> dict[str, Any]:
        """Describe a node: its attributes plus inbound/outbound connections."""
        node = self._resolve_fuzzy(name)
        if node is None:
            return {"found": False, "error": f"Node not found: {name}"}
        data = dict(self.graph.nodes[node])
        out_edges = [
            {"node": v, "title": self.graph.nodes[v].get("title", v),
             "edge_type": d.get("edge_type", "links_to")}
            for _, v, d in self.graph.out_edges(node, data=True)
        ]
        in_edges = [
            {"node": u, "title": self.graph.nodes[u].get("title", u),
             "edge_type": d.get("edge_type", "links_to")}
            for u, _, d in self.graph.in_edges(node, data=True)
        ]
        return {
            "found": True,
            "id": node,
            "title": data.get("title", node),
            "node_type": data.get("node_type", ""),
            "degree": self.graph.degree(node),
            "outbound": out_edges,
            "inbound": in_edges,
        }

    def chunks_for_node(self, name: str, limit: int = 5) -> list[dict]:
        """Return chunks that reference the given node."""
        node = self._resolve_fuzzy(name)
        if node is None:
            return []
        results = []
        for u, _, d in self.graph.in_edges(node, data=True):
            if d.get("edge_type") != "references":
                continue
            data = self.graph.nodes[u]
            results.append(
                {
                    "doc_hash": data.get("doc_hash", ""),
                    "doc_name": data.get("doc_name", ""),
                    "page_number": data.get("page_number", 0),
                }
            )
            if len(results) >= limit:
                break
        return results

    def score_pages(self, question: str, max_nodes: int = 15) -> list[str]:
        """
        Rank wiki pages for a question using seed matching and neighborhood
        expansion over the graph. Returns page ids (relative paths).
        """
        tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
        candidates = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in PAGE_NODE_TYPES
        ]
        if not tokens or not candidates:
            return []

        scores: dict[str, float] = {}
        for node in candidates:
            data = self.graph.nodes[node]
            title = data.get("title", "").lower()
            stem = Path(node).stem.lower()
            score = 0.0
            for tok in tokens:
                weight = 2.0 if len(tok) > 3 else 1.0
                if tok == title or tok == stem:
                    score += weight * 3
                elif tok in title or tok in stem:
                    score += weight
            if score > 0:
                scores[node] = score

        if not scores:
            return []

        # Expand top seeds by one hop (undirected) to pull in related pages
        undirected = self.graph.to_undirected()
        seeds = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)[:5]
        for seed in seeds:
            for neighbor in undirected.neighbors(seed):
                if self.graph.nodes[neighbor].get("node_type") not in PAGE_NODE_TYPES:
                    continue
                if neighbor not in scores:
                    scores[neighbor] = scores[seed] * 0.25
                else:
                    scores[neighbor] += scores[seed] * 0.25

        # Degree bonus: well-connected pages are better synthesis anchors
        for node in scores:
            scores[node] += min(self.graph.degree(node), 20) * 0.05

        ranked = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        return ranked[:max_nodes]

    def subgraph_for_query(self, question: str, max_nodes: int = 15) -> dict[str, Any]:
        """Return the ranked subgraph (nodes + edges) for a question."""
        ranked = self.score_pages(question, max_nodes=max_nodes)
        nodes = [
            {"id": n, "title": self.graph.nodes[n].get("title", n),
             "node_type": self.graph.nodes[n].get("node_type", "")}
            for n in ranked
        ]
        node_set = set(ranked)
        edges = [
            {"from": u, "to": v, "edge_type": d.get("edge_type", "links_to")}
            for u, v, d in self.graph.edges(data=True)
            if u in node_set and v in node_set
        ]
        return {"question": question, "nodes": nodes, "edges": edges}

    def hubs(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Return the most-connected non-chunk nodes."""
        ranked = sorted(
            (
                (n, self.graph.degree(n))
                for n, d in self.graph.nodes(data=True)
                if d.get("node_type") in PAGE_NODE_TYPES + ("stub",)
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"id": n, "title": self.graph.nodes[n].get("title", n),
             "node_type": self.graph.nodes[n].get("node_type", ""), "degree": deg}
            for n, deg in ranked[:top_n]
        ]

    def status(self) -> dict[str, Any]:
        """Return summary statistics about the graph."""
        node_counts: dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("node_type", "unknown")
            node_counts[t] = node_counts.get(t, 0) + 1
        edge_counts: dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            t = d.get("edge_type", "unknown")
            edge_counts[t] = edge_counts.get(t, 0) + 1
        return {
            "collection": self.collection_name,
            "graph_path": str(self.graph_path),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "hubs": self.hubs(5),
        }

    # ────────────────────────── visualization ───────────────────────

    def export_html(self, path: str | Path | None = None) -> Path:
        """Export an interactive pyvis HTML visualization of the graph."""
        from pyvis.network import Network

        out_path = Path(path) if path else self.wiki_dir / "graph.html"

        net = Network(
            height="900px",
            width="100%",
            directed=True,
            notebook=False,
            cdn_resources="in_line",
        )
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=95)

        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            degree = self.graph.degree(node)
            net.add_node(
                node,
                label=data.get("title", node),
                title=f"{data.get('title', node)} [{node_type}] degree={degree}",
                color=NODE_COLORS.get(node_type, "#666666"),
                size=8 + min(degree * 2, 30),
                shape="dot" if node_type != "chunk" else "box",
            )
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "links_to")
            net.add_edge(u, v, title=edge_type, color=EDGE_COLORS.get(edge_type, "#9AA1B0"))

        net.write_html(str(out_path))
        logger.info(f"Exported graph visualization -> {out_path}")
        return out_path
