"""
Main Textual App for Libby D. Bot.

Manages global state, screen navigation, and background workers.
"""

from textual.app import App, ComposeResult
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
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+d", "switch_screen('dashboard')", "Dashboard"),
        ("ctrl+c", "switch_screen('chat')", "Chat"),
        ("ctrl+e", "switch_screen('embed')", "Embed"),
        ("ctrl+w", "switch_screen('wiki_browser')", "Wiki"),
        ("ctrl+s", "switch_screen('settings')", "Settings"),
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

    def action_quit(self) -> None:
        self.exit()
