"""
Textual TUI entry point for Libby D. Bot.

Launched via: uv run libby
"""

import sys

from libbydbot.tui.app import LibbyApp


def run() -> None:
    app = LibbyApp()
    app.run()


if __name__ == "__main__":
    run()
