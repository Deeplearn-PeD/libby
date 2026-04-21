"""
Chat screen — interactive Q&A with Libby.
"""

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Markdown, Select, Static

from libbydbot.tui.widgets.status_bar import StatusBar


class ChatMessage(Static):
    """A single chat bubble."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.update_content(content)

    def update_content(self, content: str) -> None:
        if self.role == "user":
            self.update(f"**You:** {content}")
            self.add_class("user-message")
        else:
            self.update(content)
            self.add_class("assistant-message")


class ChatScreen(Screen):
    """Interactive chat with Libby."""

    NAME = "chat"
    BINDINGS = [
        ("ctrl+d", "switch_screen('dashboard')", "Dashboard"),
    ]

    MODES = [
        ("RAG Answer", "rag"),
        ("Free Generate", "generate"),
        ("Wiki Query", "wiki"),
    ]

    def compose(self):
        yield Static("Chat", id="title")
        with Vertical(id="chat-container"):
            with Vertical(id="chat-history"):
                yield Static("Welcome to Libby! Ask me anything about your documents.", id="chat-welcome")
            with Horizontal(id="chat-input-row"):
                yield Select(self.MODES, value="rag", id="chat-mode", prompt="Mode")
                yield Input(placeholder="Type your question and press Enter...", id="chat-input")
                yield Button("Send", id="btn-send", variant="primary")
        yield StatusBar()

    def on_mount(self):
        self._load_history()

    def _load_history(self):
        """Load recent chat history from the database."""
        try:
            libby = self.app.libby
            if libby and libby.history:
                history = libby.history.recall(user_id=1)
                # Show last 20 messages
                for mem in history[-20:]:
                    self._add_message("user", mem.question)
                    self._add_message("assistant", mem.response)
        except Exception:
            pass

    def _add_message(self, role: str, content: str) -> None:
        history = self.query_one("#chat-history", Vertical)
        msg = ChatMessage(role, content)
        history.mount(msg)
        msg.scroll_visible()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._send_message()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-send":
            self._send_message()

    def _send_message(self) -> None:
        input_widget = self.query_one("#chat-input", Input)
        question = input_widget.value.strip()
        if not question:
            return

        input_widget.value = ""
        input_widget.disabled = True
        self.query_one("#btn-send", Button).disabled = True

        self._add_message("user", question)
        thinking = Static("_Thinking..._", id="thinking")
        self.query_one("#chat-history", Vertical).mount(thinking)

        mode = self.query_one("#chat-mode", Select).value
        self.run_worker(self._get_response_worker(question, mode, thinking), thread=True)

    def _get_response_worker(self, question: str, mode: str, thinking_widget: Static):
        """Background worker for LLM calls."""
        try:
            libby = self.app.libby
            if not libby:
                response = "Libby is not initialized. Check settings and database connection."
            elif mode == "rag":
                response = libby.answer(question, collection_name=self.app.current_collection)
            elif mode == "generate":
                response = libby.generate(prompt=question)
            elif mode == "wiki":
                from libbydbot.brain.wiki import WikiManager
                wiki = WikiManager(
                    collection_name=self.app.current_collection,
                    wiki_base=self.app._settings.wiki_base_path if self.app._settings else "",
                    model=self.app.current_model,
                )
                result = wiki.query(question, file_answer=False)
                response = f"**Confidence:** {result['confidence']}\n\n{result['answer']}"
                if result.get("sources_used"):
                    response += "\n\n**Sources:** " + ", ".join(result["sources_used"])
            else:
                response = libby.ask(question)
        except Exception as e:
            response = f"Error: {e}"

        self.app.call_from_thread(self._on_response_ready, thinking_widget, response)

    def _on_response_ready(self, thinking_widget: Static, response: str) -> None:
        thinking_widget.remove()
        self._add_message("assistant", response)
        input_widget = self.query_one("#chat-input", Input)
        input_widget.disabled = False
        self.query_one("#btn-send", Button).disabled = False
        input_widget.focus()
