"""
Status bar widget showing current system state.
"""

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Bottom status bar reactive to app state."""

    def on_mount(self):
        self.update_content()

    def watch_app_state(self):
        self.update_content()

    def update_content(self):
        app = self.app
        collection = getattr(app, "current_collection", "main")
        model = getattr(app, "current_model", "unknown")
        status = getattr(app, "status_message", "Ready")
        self.update(
            f" Collection: {collection} | Model: {model} | {status} "
            f"| [b]Ctrl+Q[/b] to exit "
        )

    def on_app_current_collection_changed(self, event):
        self.update_content()

    def on_app_current_model_changed(self, event):
        self.update_content()

    def on_app_status_message_changed(self, event):
        self.update_content()
