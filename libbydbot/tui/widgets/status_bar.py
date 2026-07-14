"""
Status bar widget showing current system state and navigation hints.

Sits as a single bottom-docked widget (two rows) so it never collides with
other docked widgets:

* Row 1 — live state (collection / model / status) plus the global navigation
  keys that work on every screen.
* Row 2 — the key bindings declared by the currently active screen, so the
  user always knows how to get around (including back to the dashboard).
"""

from textual.widgets import Static


class StatusBar(Static):
    """Bottom status bar reactive to app state."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 2;
        background: $surface;
        color: $text;
        padding: 0 2;
    }
    """

    def on_mount(self):
        self.update_content()

    def watch_app_state(self):
        self.update_content()

    # Keys that are always shown in the global-nav row, so we skip them when
    # listing per-screen bindings to avoid duplicates.
    GLOBAL_KEYS = {"ctrl+d", "ctrl+q"}

    def _screen_hints(self) -> str:
        """Build a hint string from the active screen's declared bindings."""
        screen = getattr(self.app, "screen", None)
        bindings = getattr(screen, "BINDINGS", []) or []
        hints = []
        for binding in bindings:
            # Bindings may be tuples (key, action, description[, show]) or
            # Binding dataclass-like objects.
            if isinstance(binding, (tuple, list)):
                key = binding[0] if len(binding) > 0 else None
                desc = binding[2] if len(binding) > 2 else None
            else:
                key = getattr(binding, "key", None)
                desc = getattr(binding, "description", None)
            if not key or not desc:
                continue
            if key.lower() in self.GLOBAL_KEYS:
                continue
            hints.append(f"[b]{key.upper()}[/b] {desc}")
        return "  ".join(hints)

    def update_content(self):
        app = self.app
        collection = getattr(app, "current_collection", "main")
        model = getattr(app, "current_model", "unknown")
        status = getattr(app, "status_message", "Ready")
        global_nav = "[b]CTRL+D[/b] Dashboard   [b]CTRL+Q[/b] Quit"
        screen_nav = self._screen_hints()
        self.update(
            f" Collection: {collection} | Model: {model} | {status}\n"
            f" {global_nav}   {screen_nav}"
        )

    def on_app_current_collection_changed(self, event):
        self.update_content()

    def on_app_current_model_changed(self, event):
        self.update_content()

    def on_app_status_message_changed(self, event):
        self.update_content()
