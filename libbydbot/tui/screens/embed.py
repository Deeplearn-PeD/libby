"""
Embed screen — ingest and embed documents with visual progress.
"""

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DirectoryTree, Input, RichLog, Static

from libbydbot.tui.widgets.status_bar import StatusBar


class EmbedScreen(Screen):
    """Embed documents from the filesystem."""

    NAME = "embed"
    BINDINGS = [
        ("ctrl+d", "switch_screen('dashboard')", "Dashboard"),
    ]

    def compose(self):
        yield Static("Embed Documents", id="title")
        with Vertical(id="embed-container"):
            with Horizontal(id="embed-top"):
                with Vertical(id="embed-left"):
                    yield Static("Select folder:")
                    yield DirectoryTree(".", id="embed-tree")
                with Vertical(id="embed-right"):
                    yield Static("Settings")
                    yield Input(value=self.app.current_collection, placeholder="Collection name", id="embed-collection")
                    yield Button("Start Embedding", id="btn-embed-start", variant="success")
                    yield Button("Stop", id="btn-embed-stop", variant="error", disabled=True)
            with Vertical(id="embed-progress"):
                yield Static("Progress")
                yield RichLog(id="embed-log", wrap=True)
        yield StatusBar()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._selected_path = str(event.path)
        self.query_one("#embed-log", RichLog).write(f"Selected: {self._selected_path}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-embed-start":
            self._start_embedding()
        elif btn_id == "btn-embed-stop":
            self._stop_embedding()

    def _start_embedding(self) -> None:
        if not hasattr(self, "_selected_path"):
            self.notify("Please select a folder first.", severity="warning")
            return

        collection = self.query_one("#embed-collection", Input).value or self.app.current_collection
        self.app.current_collection = collection

        self.query_one("#btn-embed-start", Button).disabled = True
        self.query_one("#btn-embed-stop", Button).disabled = False

        log = self.query_one("#embed-log", RichLog)
        log.clear()
        log.write(f"Starting embedding of: {self._selected_path}")
        log.write(f"Collection: {collection}")

        self.run_worker(
            lambda: self._embed_worker(self._selected_path, collection, log),
            thread=True,
        )

    def _stop_embedding(self) -> None:
        self.notify("Stopping... (current file will finish)")
        self.query_one("#btn-embed-start", Button).disabled = False
        self.query_one("#btn-embed-stop", Button).disabled = True

    def _set_button_state(self, start_disabled: bool, stop_disabled: bool) -> None:
        """Update button disabled states (safe to call from worker threads)."""
        self.query_one("#btn-embed-start", Button).disabled = start_disabled
        self.query_one("#btn-embed-stop", Button).disabled = stop_disabled

    def _safe_call(self, fn, *args, **kwargs):
        """Call fn directly if on the main thread, otherwise via call_from_thread."""
        import threading

        if threading.get_ident() == self.app._thread_id:
            fn(*args, **kwargs)
        else:
            self.app.call_from_thread(fn, *args, **kwargs)

    def _embed_worker(self, path: str, collection: str, log: RichLog):
        def progress_callback(doc_name, chunk_idx, total):
            msg = f"  {doc_name} — chunk {chunk_idx + 1}/{total}"
            self._safe_call(log.write, msg)

        try:
            libby = self.app.libby
            if not libby:
                self._safe_call(log.write, "Error: Libby not initialized.")
                return

            libby.DE.collection_name = collection
            libby.DE.embed_path(path, callback=progress_callback)
            self._safe_call(log.write, "Embedding complete!")
            self._safe_call(self.app.notify, "Embedding complete!")
        except Exception as e:
            self._safe_call(log.write, f"Error: {e}")
            self._safe_call(self.app.notify, f"Embedding failed: {e}", severity="error")
        finally:
            self._safe_call(self._set_button_state, False, True)
