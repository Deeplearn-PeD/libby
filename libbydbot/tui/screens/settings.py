"""
Settings screen — view and tweak runtime configuration.
"""

from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Select, Static

from libbydbot.tui.widgets.status_bar import StatusBar


class SettingsScreen(Screen):
    """View and modify Libby settings."""

    NAME = "settings"
    BINDINGS = [
        ("ctrl+d", "switch_screen('dashboard')", "Dashboard"),
    ]

    def compose(self):
        yield Static("Settings", id="title")
        with Vertical(id="settings-form"):
            yield Static("LLM Model")
            model_options = self._build_model_options()
            yield Select(model_options, value=self._get_current_model_value(model_options), id="setting-model")

            yield Static("Embedding Model")
            embed_options = self._build_embed_options()
            yield Select(embed_options, value=self._get_current_embed_value(embed_options), id="setting-embed-model")

            yield Static("Collection")
            yield Input(value=self.app.current_collection, id="setting-collection")

            yield Static("Database URL (read-only)")
            dburl = "postgresql://libby:libby123@localhost:5432/libby"
            if self.app._settings and hasattr(self.app._settings, "pgurl"):
                dburl = str(self.app._settings.pgurl)
            yield Input(value=dburl, disabled=True, id="setting-dburl")

            yield Static("Wiki Base Path (read-only)")
            wiki_base = self.app._settings.wiki_base_path if self.app._settings else "~/.libby/wikis"
            yield Input(value=wiki_base, disabled=True, id="setting-wiki-base")

            yield Button("Apply", id="btn-apply", variant="primary")
        yield StatusBar()

    def _build_model_options(self):
        settings = self.app._settings
        if settings and settings.models:
            return [(name, details["code"]) for name, details in settings.models.items()]
        return [
            ("Llama 3.2", "llama3.2"),
            ("Gemma 3", "gemma3"),
            ("GPT-4o", "gpt-4o"),
            ("Qwen 3", "qwen3"),
            ("Gemini", "gemini"),
        ]

    def _get_current_model_value(self, options):
        current = self.app.current_model
        for label, value in options:
            if value == current:
                return current
        # If current model not in options, return first option
        return options[0][1] if options else Select.BLANK

    def _build_embed_options(self):
        settings = self.app._settings
        if settings and settings.embedding_models:
            return [(name, details["code"]) for name, details in settings.embedding_models.items()]
        return [
            ("Gemma Embedding", "embeddinggemma"),
            ("Mxbai", "mxbai-embed-large"),
            ("Gemini", "gemini-embedding-001"),
        ]

    def _get_current_embed_value(self, options):
        settings = self.app._settings
        current = settings.default_embedding_model if settings else "embeddinggemma"
        for label, value in options:
            if value == current:
                return current
        return options[0][1] if options else Select.BLANK

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-apply":
            model = self.query_one("#setting-model", Select).value
            collection = self.query_one("#setting-collection", Input).value
            if model and model != Select.BLANK:
                self.app.current_model = model
            if collection:
                self.app.current_collection = collection
            self.notify("Settings applied. Libby will reconnect on next operation.")
