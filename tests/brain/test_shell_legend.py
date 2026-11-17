"""
Tests for the Libby graph shell legend overlay.

The rendered shell (full + ego) embeds a `kg-legend` overlay listing
node types (mirroring NODE_COLORS), topic communities with their chip
colors, and the cross-community bridge count — so users can identify
what the colors mean without leaving the canvas.

Run inside the Libby repo: uv run pytest tests/brain/test_shell_legend.py -v
"""

from pathlib import Path

import pytest

from libbydbot.brain.graph import (
    LEGEND_NODE_TYPES,
    WikiKnowledgeGraph,
    _community_chip_color,
    _legend_html,
    render_shell_html,
)


def test_legend_node_types_match_node_colors():
    for node_type, color in LEGEND_NODE_TYPES:
        assert node_type in ("source", "entity", "concept", "synthesis", "stub")
        assert color.startswith("#")


def test_legend_html_contains_node_types():
    html = _legend_html({})
    assert 'id="kg-legend"' in html
    for node_type in ("source", "entity", "concept", "synthesis"):
        assert f">{node_type}</div>" in html


def test_legend_html_communities_and_bridges():
    html = _legend_html(
        community_labels={"0": "Vector control", "1": "Vaccination"},
        bridges=[
            {
                "from": "a", "to": "b",
                "from_title": "Dengue", "to_title": "Climate",
                "from_community": "Vector control",
                "to_community": "Climate models",
            }
        ],
    )
    assert "Vector control" in html
    assert "Vaccination" in html
    assert "1 cross-community bridge(s)" in html


def test_legend_html_empty_communities_omits_section():
    html = _legend_html({})
    assert "Communities" not in html


def test_chip_color_deterministic():
    assert _community_chip_color("abc") == _community_chip_color("abc")
    assert _community_chip_color("abc") != _community_chip_color("xyz")


def test_rendered_shell_embeds_legend():
    wiki = Path(__file__).parent / "_tmp_shell_legend_wiki"
    wiki.mkdir(parents=True, exist_ok=True)
    for sub in ("sources", "entities", "concepts", "synthesis"):
        (wiki / sub).mkdir(exist_ok=True)
    html = render_shell_html(boot={"ego": False}, legend_html=_legend_html({}))
    assert 'id="kg-legend"' in html
    import shutil

    shutil.rmtree(wiki, ignore_errors=True)
