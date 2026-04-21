"""
Wiki Ingest screen — feed documents into the wiki.
"""

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Input, RichLog, Static

from libbydbot.tui.widgets.status_bar import StatusBar


class WikiIngestScreen(Screen):
    """Ingest documents into the markdown wiki."""

    NAME = "wiki_ingest"
    BINDINGS = [
        ("ctrl+d", "switch_screen('dashboard')", "Dashboard"),
        ("ctrl+w", "switch_screen('wiki_browser')", "Wiki Browser"),
    ]

    def compose(self):
        yield Static("Wiki Ingest", id="title")
        with Vertical(id="embed-container"):
            with Horizontal(id="embed-top"):
                with Vertical(id="embed-left"):
                    yield Static("Select folder with PDFs:")
                    yield DirectoryTree(".", id="ingest-tree")
                with Vertical(id="embed-right"):
                    yield Static("Settings")
                    yield Input(value=self.app.current_collection, placeholder="Collection", id="ingest-collection")
                    yield Button("Start Ingest", id="btn-ingest-start", variant="success")
            with Vertical(id="embed-progress"):
                yield Static("Progress")
                yield RichLog(id="ingest-log", wrap=True)
        yield StatusBar()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._selected_path = str(event.path)
        self.query_one("#ingest-log", RichLog).write(f"Selected: {self._selected_path}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-ingest-start":
            self._start_ingest()

    def _start_ingest(self) -> None:
        if not hasattr(self, "_selected_path"):
            self.notify("Please select a folder first.", severity="warning")
            return

        collection = self.query_one("#ingest-collection", Input).value or self.app.current_collection
        self.app.current_collection = collection

        log = self.query_one("#ingest-log", RichLog)
        log.clear()
        log.write(f"Starting wiki ingest from: {self._selected_path}")
        log.write(f"Collection: {collection}")

        self.run_worker(self._ingest_worker(self._selected_path, collection, log), thread=True)

    def _ingest_worker(self, path: str, collection: str, log: RichLog):
        import os
        import fitz
        from glob import glob

        try:
            from libbydbot.brain.wiki import WikiManager
            wiki = WikiManager(
                collection_name=collection,
                wiki_base=self.app._settings.wiki_base_path if self.app._settings else "",
                model=self.app.current_model,
            )

            pdf_paths = glob(os.path.join(path, "*.pdf"))
            if not pdf_paths:
                self.app.call_from_thread(log.write, "No PDF files found.")
                return

            for pdf_path in pdf_paths:
                try:
                    doc = fitz.open(pdf_path)
                    text = ""
                    for page in doc:
                        text += page.get_text() + "\n"
                    doc_name = doc.metadata.get("title", os.path.basename(pdf_path))
                except Exception as e:
                    self.app.call_from_thread(log.write, f"Error reading {pdf_path}: {e}")
                    continue

                self.app.call_from_thread(log.write, f"Ingesting: {doc_name}...")
                result = wiki.ingest_source(doc_name, text)
                self.app.call_from_thread(
                    log.write,
                    f"  Done: {result['pages_touched']} pages touched, "
                    f"{result['entities_created']} entities, {result['concepts_created']} concepts",
                )

            self.app.call_from_thread(log.write, "Wiki ingest complete!")
            self.app.call_from_thread(self.app.notify, "Wiki ingest complete!")
        except Exception as e:
            self.app.call_from_thread(log.write, f"Error: {e}")
            self.app.call_from_thread(self.app.notify, f"Ingest failed: {e}", severity="error")
