"""
Tests for the Textual TUI.
Uses Textual's built-in testing harness with pytest-asyncio.
"""

import pytest
import pytest_asyncio

from libbydbot.tui.app import LibbyApp
from libbydbot.tui.screens import (
    ChatScreen,
    DashboardScreen,
    EmbedScreen,
    SettingsScreen,
    WikiBrowserScreen,
)


@pytest_asyncio.fixture
async def pilot():
    app = LibbyApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        yield pilot, app


class TestAppCreation:
    @pytest.mark.asyncio
    async def test_app_mounts(self, pilot):
        p, app = pilot
        assert app.is_running
        assert app.current_collection == "main"

    @pytest.mark.asyncio
    async def test_dashboard_screen(self, pilot):
        p, app = pilot
        assert isinstance(app.screen, DashboardScreen)
        dashboard = app.screen.query_one("#dashboard-grid")
        assert dashboard is not None

    @pytest.mark.asyncio
    async def test_navigate_to_chat(self, pilot):
        p, app = pilot
        await p.click("#btn-chat")
        await p.pause()
        assert isinstance(app.screen, ChatScreen)

    @pytest.mark.asyncio
    async def test_navigate_to_settings(self, pilot):
        p, app = pilot
        # Click on settings button; note: button may need direct interaction
        btn = app.screen.query_one("#btn-settings")
        assert btn is not None
        btn.press()
        await p.pause()
        assert isinstance(app.screen, SettingsScreen)

    @pytest.mark.asyncio
    async def test_navigate_to_wiki(self, pilot):
        p, app = pilot
        await p.click("#btn-wiki")
        await p.pause()
        assert isinstance(app.screen, WikiBrowserScreen)

    @pytest.mark.asyncio
    async def test_navigate_to_embed(self, pilot):
        p, app = pilot
        await p.click("#btn-embed")
        await p.pause()
        assert isinstance(app.screen, EmbedScreen)

    @pytest.mark.asyncio
    async def test_keyboard_shortcuts(self, pilot):
        p, app = pilot
        assert isinstance(app.screen, DashboardScreen)

        await p.press("ctrl+c")
        await p.pause()
        assert isinstance(app.screen, ChatScreen)

        await p.press("ctrl+e")
        await p.pause()
        assert isinstance(app.screen, EmbedScreen)

        await p.press("ctrl+w")
        await p.pause()
        assert isinstance(app.screen, WikiBrowserScreen)

        await p.press("ctrl+s")
        await p.pause()
        assert isinstance(app.screen, SettingsScreen)

        await p.press("ctrl+d")
        await p.pause()
        assert isinstance(app.screen, DashboardScreen)


class TestReactiveState:
    @pytest.mark.asyncio
    async def test_collection_change(self, pilot):
        p, app = pilot
        app.current_collection = "test_coll"
        assert app.current_collection == "test_coll"
        assert app._libby is None  # Should reset on change

    @pytest.mark.asyncio
    async def test_model_change(self, pilot):
        p, app = pilot
        app.current_model = "gemma3"
        assert app.current_model == "gemma3"
        assert app._libby is None  # Should reset on change
