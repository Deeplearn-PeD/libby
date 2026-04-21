"""
Wiki Browser screen — navigate and read wiki pages.
"""

from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Markdown, Static, Tree

from libbydbot.tui.widgets.status_bar import StatusBar


class WikiBrowserScreen(Screen):
    """Browse the markdown wiki with a file tree and markdown viewer."""

    NAME = "wiki_browser"
    BINDINGS = [
        ("ctrl+d", "switch_screen('dashboard')", "Dashboard"),
        ("i", "switch_screen('wiki_ingest')", "Ingest"),
        ("r", "action_refresh", "Refresh"),
    ]

    def compose(self):
        yield Static("Wiki Browser", id="title")
        with Horizontal(id="wiki-controls"):
            yield Input(value=self.app.current_collection, placeholder="Collection", id="wiki-collection")
            yield Button("Refresh", id="btn-refresh", variant="primary")
            yield Button("Ingest", id="btn-ingest", variant="success")
            yield Button("Lint", id="btn-lint", variant="warning")
        with Horizontal(id="wiki-container"):
            yield Tree("Wiki", id="wiki-tree")
            yield Markdown("Select a page to view.", id="wiki-viewer")
        yield StatusBar()

    def on_mount(self):
        self._load_wiki_tree()

    def _load_wiki_tree(self):
        tree = self.query_one("#wiki-tree", Tree)
        tree.clear()
        tree.root.expand()

        try:
            from libbydbot.brain.wiki import WikiManager
            wiki = WikiManager(
                collection_name=self.app.current_collection,
                wiki_base=self.app._settings.wiki_base_path if self.app._settings else "",
                model=self.app.current_model,
            )
            self._wiki = wiki
            root = tree.root

            # Add index and log
            root.add_leaf("index.md", data=wiki.index_path)
            root.add_leaf("log.md", data=wiki.log_path)

            # Add directories
            for label, directory in [
                ("Sources", wiki.sources_dir),
                ("Entities", wiki.entities_dir),
                ("Concepts", wiki.concepts_dir),
                ("Synthesis", wiki.synthesis_dir),
            ]:
                node = root.add(label, expand=True)
                if directory.exists():
                    for md_file in sorted(directory.glob("*.md")):
                        node.add_leaf(md_file.stem, data=md_file)
        except Exception as e:
            self.notify(f"Could not load wiki: {e}", severity="error")

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node
        if node.data and isinstance(node.data, Path):
            try:
                content = node.data.read_text(encoding="utf-8")
                viewer = self.query_one("#wiki-viewer", Markdown)
                viewer.update(content)
            except Exception as e:
                self.notify(f"Error reading page: {e}", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-refresh":
            self.app.current_collection = self.query_one("#wiki-collection", Input).value or "main"
            self._load_wiki_tree()
        elif btn_id == "btn-ingest":
            self.app.push_screen("wiki_ingest")
        elif btn_id == "btn-lint":
            self._run_lint()

    def action_refresh(self):
        self._load_wiki_tree()

    def _run_lint(self):
        self.notify("Running wiki lint...")
        self.run_worker(self._lint_worker(), thread=True)

    def _lint_worker(self):
        try:
            from libbydbot.brain.wiki import WikiManager
            wiki = WikiManager(
                collection_name=self.app.current_collection,
                wiki_base=self.app._settings.wiki_base_path if self.app._settings else "",
                model=self.app.current_model,
            )
            report = wiki.lint(auto_fix=False)
            msg = (
                f"Lint complete:\n"
                f"  Orphans: {len(report['orphan_pages'])}\n"
                f"  Broken links: {len(report['broken_links'])}\n"
                f"  Contradictions: {len(report['contradictions'])}\n"
                f"  Missing pages: {len(report['missing_pages'])}"
            )
            self.app.call_from_thread(self.app.notify, msg)
        except Exception as e:
            self.app.call_from_thread(self.app.notify, f"Lint failed: {e}", severity="error")
