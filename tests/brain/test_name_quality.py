"""
Tests for name-quality validation (brain/name_quality.py).

Generic names — single letters, pure numbers/dates, session/chat
identifiers — carry no semantic value and must be filtered from wiki
pages and knowledge-graph nodes. Legitimate short names (AI, SP, RJ)
and real terms (Dengue, COVID-19) must survive.

Run with: uv run pytest tests/brain/test_name_quality.py -v
"""

import pytest

from libbydbot.brain.name_quality import has_semantic_value


@pytest.mark.parametrize(
    "name",
    [
        "A",                    # single letter (multiple-choice answer)
        "B",
        "7",                    # single digit
        "123",                  # pure number
        "12.5",                 # decimal
        "2025-01-15",           # date
        "1,234 / 567",          # numeric soup
        "session-123",          # session identifier
        "Session 42",           # session identifier (words, spaced)
        "CHAT-7",               # session identifier (other prefix, upper)
        "transcript_9",         # transcript identifier
        "-",                    # punctuation
        "_",
        "  ",                   # whitespace only
        "",                     # empty
    ],
)
def test_generic_names_rejected(name):
    assert not has_semantic_value(name), f"{name!r} should be rejected"


@pytest.mark.parametrize(
    "name",
    [
        "Dengue",
        "Zika vaccine",
        "AI",                   # two letters, legitimate
        "SP",                   # state abbreviation
        "RJ",
        "COVID-19",             # word core + digits
        "SUS",                  # acronym
        "Sergipe state",
        "Model 2.1 evaluation", # word core with version digits
    ],
)
def test_semantic_names_accepted(name):
    assert has_semantic_value(name), f"{name!r} should be accepted"
