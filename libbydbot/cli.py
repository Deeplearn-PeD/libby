"""
CLI entry point for Libby D. Bot.

By default, launches the Textual TUI.
For scripting, use the `libby-cli` command instead.
"""

from libbydbot.tui.app import LibbyApp


def main():
    """Launch the Libby TUI."""
    app = LibbyApp()
    app.run()


if __name__ == "__main__":
    main()
