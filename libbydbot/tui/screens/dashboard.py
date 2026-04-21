"""
Dashboard screen — landing page with collection overview and quick actions.
"""

from textual.containers import Grid, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Static

from libbydbot.tui.widgets.status_bar import StatusBar


class DashboardScreen(Screen):
    """Main dashboard showing collections and quick actions."""

    NAME = "dashboard"
    BINDINGS = [
        ("c", "switch_screen('chat')", "Chat"),
        ("e", "switch_screen('embed')", "Embed"),
        ("w", "switch_screen('wiki_browser')", "Wiki"),
        ("s", "switch_screen('settings')", "Settings"),
    ]

    def compose(self):
        yield Static("Dashboard", id="title")
        with Grid(id="dashboard-grid"):
            with Vertical(id="collections-panel"):
                yield Static("Collections", classes="panel-title")
                yield DataTable(id="collections-table", show_cursor=True)
                yield Static("Select a collection to activate it.", id="collection-hint")
            with Vertical(id="actions-panel"):
                yield Static("Quick Actions", classes="panel-title")
                yield Button("Chat (Ctrl+C)", id="btn-chat", variant="primary")
                yield Button("Embed Documents (Ctrl+E)", id="btn-embed", variant="success")
                yield Button("Browse Wiki (Ctrl+W)", id="btn-wiki", variant="warning")
                yield Button("Settings (Ctrl+S)", id="btn-settings", variant="default")
                yield Button("Ingest to Wiki", id="btn-wiki-ingest", variant="default")
        yield StatusBar()

    def on_mount(self):
        table = self.query_one("#collections-table", DataTable)
        table.add_columns("Collection", "Documents", "Embedding Model")
        self._refresh_collections()

    def _refresh_collections(self):
        table = self.query_one("#collections-table", DataTable)
        table.clear()
        try:
            libby = self.app.libby
            if libby and libby.DE:
                docs = libby.DE.get_embedded_documents()
                from collections import Counter
                counts = Counter(col for _, col in docs)
                info = libby.DE.get_embedding_model_info()
                model_map: dict[str, str] = {}
                for model, collections in info.get("models", {}).items():
                    for coll, _ in collections.items():
                        model_map[coll] = model
                for coll, count in counts.items():
                    table.add_row(coll, str(count), model_map.get(coll, "unknown"))
        except Exception as e:
            table.add_row("(error)", str(e), "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-chat":
            self.app.switch_screen("chat")
        elif btn_id == "btn-embed":
            self.app.switch_screen("embed")
        elif btn_id == "btn-wiki":
            self.app.switch_screen("wiki_browser")
        elif btn_id == "btn-settings":
            self.app.switch_screen("settings")
        elif btn_id == "btn-wiki-ingest":
            self.app.switch_screen("wiki_ingest")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one("#collections-table", DataTable)
        row = table.get_row(event.row_key)
        if row:
            collection = row[0]
            self.app.current_collection = collection
            self.notify(f"Active collection: {collection}")
