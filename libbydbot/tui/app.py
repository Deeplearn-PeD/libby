"""
Main Textual App for Libby D. Bot.

Manages global state, screen navigation, and background workers.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from libbydbot.settings import Settings
from libbydbot.tui.screens import (
    ChatScreen,
    DashboardScreen,
    EmbedScreen,
    SettingsScreen,
    WikiBrowserScreen,
    WikiIngestScreen,
)
from libbydbot.tui.widgets.status_bar import StatusBar


class LibbyApp(App):
    """Libby D. Bot Terminal User Interface."""

    CSS_PATH = "libby.tcss"
    # Priority bindings so navigation works even when a widget (e.g. an Input)
    # would otherwise capture the key (Ctrl+D, Ctrl+E, etc. are readline keys
    # consumed by focused Input widgets).
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+d", "switch_screen('dashboard')", "Dashboard", priority=True),
        Binding("ctrl+c", "switch_screen('chat')", "Chat", priority=True),
        Binding("ctrl+e", "switch_screen('embed')", "Embed", priority=True),
        Binding("ctrl+w", "switch_screen('wiki_browser')", "Wiki", priority=True),
        Binding("ctrl+s", "switch_screen('settings')", "Settings", priority=True),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "chat": ChatScreen,
        "embed": EmbedScreen,
        "wiki_browser": WikiBrowserScreen,
        "wiki_ingest": WikiIngestScreen,
        "settings": SettingsScreen,
    }

    # Global reactive state
    current_collection: reactive[str] = reactive("main")
    current_model: reactive[str] = reactive("llama3.2")
    status_message: reactive[str] = reactive("Ready")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._libby = None
        self._settings = None
        try:
            self._settings = Settings()
            self.current_model = self._settings.default_model
            Path.home().joinpath(".libby", "data").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.status_message = f"Settings error: {e}"

    @property
    def libby(self):
        """Lazy-initialize LibbyInterface to avoid blocking app startup."""
        if self._libby is None:
            from libbydbot.cli_legacy import LibbyInterface

            try:
                self._libby = LibbyInterface(
                    collection_name=self.current_collection,
                    model=self.current_model,
                    dburl=self._settings.db_url,
                    embed_db=self._settings.embed_db_url,
                )
                self.status_message = f"Connected ({self.current_model})"
            except Exception as e:
                self.status_message = f"Connection failed: {e}"
                self.notify(f"Failed to initialize Libby: {e}", severity="error")
        return self._libby

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Libby D. Bot — AI Librarian", id="title")
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("dashboard")

    def watch_current_collection(self, collection: str) -> None:
        """When collection changes, reset libby instance so next access re-initializes."""
        self._libby = None
        self.status_message = f"Collection: {collection}"
        self.notify(f"Switched to collection: {collection}")

    def watch_current_model(self, model: str) -> None:
        """When model changes, reset libby instance."""
        self._libby = None
        self.status_message = f"Model: {model}"

    @property
    def wiki_model(self) -> str:
        """LLM model to use for wiki operations.

        Prefers the dedicated ``WIKI_MODEL`` setting, then the default chat
        model, then the currently selected model. Non-thinking models are
        recommended (thinking models reject pydantic-ai structured output).
        """
        if self._settings:
            return self._settings.wiki_model or self._settings.default_model or self.current_model
        return self.current_model

    def action_quit(self) -> None:
        self.exit()
