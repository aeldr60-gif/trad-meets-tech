from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


# Include common repeat/double-bar variants used in TheSession ABC bodies.
# Note: ordering/longest-match is handled in tokenize_abc.
BAR_TOKENS = {"|", "||", "|:", "||:", ":|", ":||", "::", "[|", "|]"}

# Endings like [1 [2 [3, etc. Capture all consecutive digits.
ENDING_START_RE = re.compile(r"\[\d+")
CHORD_RE = re.compile(r'"{1,2}([^"]+?)"{1,2}')

# Inline ABC field directives like [K:Dmix] or [L:1/8]. TheSession often embeds these
# in the body. Treat them as a single token so we don't mis-tokenize the directive
# value (e.g., "K:D" -> note "D").
BRACKET_FIELD_RE = re.compile(r"\[[A-Za-z]:[^\]]*\]")



@dataclass(frozen=True)
class AbcToken:
    kind: str  # note, rest, bar, chord, field, other
    text: str


_NOTE_RE = re.compile(
    r"""
    (?P<accidental>\^{1,2}|_{1,2}|=)?   # ^ ^^ _ __ =
    (?P<note>[A-Ga-g])                 # letter
    (?P<octave>[,']*)                  # octave marks
    # Length forms in TheSession bodies are usually simple (2, /, /2), but can also be
    # ratios like 3/2 or double slashes like //.
    (?P<length>\d+/\d+|\d+//|//|\d+/|/\d+|/|\d+)?
    """,
    re.VERBOSE,
)

_REST_RE = re.compile(r"(?P<rest>[zZxX])(?P<length>\d+/\d+|\d+//|//|\d+/|/\d+|/|\d+)?")

def tokenize_abc(abc: str) -> Iterator[AbcToken]:
    i = 0
    n = len(abc)

    while i < n:
        # skip whitespace/newlines
        if abc[i].isspace():
            i += 1
            continue

        # Inline field directives like [K:Dmix] or [L:1/8]
        m_field = BRACKET_FIELD_RE.match(abc, i)
        if m_field:
            yield AbcToken("field", m_field.group(0))
            i = m_field.end()
            continue

        # chord tokens like ""Am""
        m = CHORD_RE.match(abc, i)
        if m:
            yield AbcToken("chord", m.group(1).strip())
            i = m.end()
            continue

        # endings like [1 [2 [10
        m_end = ENDING_START_RE.match(abc, i)
        if m_end:
            yield AbcToken("ending", m_end.group(0))
            i = m_end.end()
            continue

        # bar / repeat tokens (longest match first)
        for bt in sorted(BAR_TOKENS, key=len, reverse=True):
            if abc.startswith(bt, i):
                yield AbcToken("bar", bt)
                i += len(bt)
                break
        else:
            # note
            m = _NOTE_RE.match(abc, i)
            if m:
                yield AbcToken("note", m.group(0))
                i = m.end()
                continue

            # rest
            m = _REST_RE.match(abc, i)
            if m:
                yield AbcToken("rest", m.group(0))
                i = m.end()
                continue

            # fallback: consume one char
            yield AbcToken("other", abc[i])
            i += 1
