"""
Name-quality validation for wiki pages and knowledge-graph nodes.

Generic names — single letters, pure numbers/dates, session identifiers
such as ``session-123`` — carry no semantic value: they are extraction
noise from LLM entity recognition over raw chat text. This module is the
single source of truth for what counts as a meaningful page/node name,
used at wiki-ingest time (skip junk entity/concept pages) and at graph
build time (prune junk nodes, including legacy ones already on disk).

Deliberate trade-off: single-letter names are rejected even though a few
are legitimate (the R language, grade letters) — in chat-derived wikis
they are overwhelmingly extraction noise.
"""

import re

#: A run of at least two consecutive letters — the minimum "word core"
#: a meaningful name must contain (accents included).
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}")

#: Numeric/date/punctuation soup with no word core.
_NUMERIC_RE = re.compile(r"^[\d\s.,:;\-/]+$")

#: Session/chat/transcript/log identifiers: "session-123", "chat 42",
#: "transcript_7", "Chat Session 123", "CHAT-7".
_GENERIC_SOURCE_RE = re.compile(
    r"^(chat|session|transcript|log)[-_ ]?\d+$", re.IGNORECASE
)


def has_semantic_value(name: str) -> bool:
    """True when a page/node name carries semantic value.

    Rejects: single characters, pure numbers/dates/punctuation, and
    session/chat identifiers. Accepts everything containing at least a
    two-letter word core (``AI``, ``SP``, ``RJ``, ``Dengue``, ``COVID-19``).
    """
    name = (name or "").strip()
    if len(name) <= 1:
        return False
    if not _WORD_RE.search(name):
        return False
    if _NUMERIC_RE.fullmatch(name):
        return False
    if _GENERIC_SOURCE_RE.match(name):
        return False
    return True
