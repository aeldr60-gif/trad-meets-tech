from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

"""
Tokenizes ABC notation into structured elements such as notes, rests, barlines,
chords, repeat/ending markers, and inline field directives. Handles TheSession-
style variants, including complex bar tokens (|:, :||, |1, etc.) and embedded
[K:], [L:], and quoted chord symbols. Produces a stream of AbcToken objects
preserving the original text for accurate downstream parsing or re emission.
"""




# Include common repeat/double-bar variants used in TheSession ABC bodies.
# We match barlines using a regex to guarantee that measure separators are
# tokenized correctly (and therefore measure_number increments).
BAR_TOKENS = {"|", "||", "|:", "||:", ":|", ":||", "::", "[|", "|]"}

# Bar regex (longest tokens first). Also supports common first/second ending
# bar forms like |1 or :|2 as single tokens so the original ABC can be re-emitted.
BAR_RE = re.compile(
    r"(\|\|:|:\|\||\|\||\|:|:\|\d+|\|\d+|:\||::|\[\||\|\]|\|)"
)

# Endings like [1 [2 [3, etc. Capture all consecutive digits.
ENDING_START_RE = re.compile(r"\[\d+")
CHORD_RE = re.compile(r'"{1,2}([^"]+?)"{1,2}')

# Inline ABC field directives like [K:Dmix] or [L:1/8]. TheSession often embeds these
# in the body. Treat them as a single token so we don't mis-tokenize the directive
# value (e.g., "K:D" -> note "D").
BRACKET_FIELD_RE = re.compile(r"\[[A-Za-z]:[^\]]*\]")


@dataclass(frozen=True)
class AbcToken:
    kind: str  # note, rest, bar, chord, field, ending, other
    text: str


_NOTE_RE = re.compile(
    r'''
    (?P<accidental>\^{1,2}|_{1,2}|=)?   # ^ ^^ _ __ =
    (?P<note>[A-Ga-g])                 # letter
    (?P<octave>[,']*)                  # octave marks
    # Length forms in TheSession bodies are usually simple (2, /, /2), but can also be
    # ratios like 3/2 or double slashes like //.
    (?P<length>\d+/\d+|\d+//|//|\d+/|/\d+|/|\d+)?
    ''',
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

        # bar / repeat tokens (regex handles longest match)
        m_bar = BAR_RE.match(abc, i)
        if m_bar:
            yield AbcToken("bar", m_bar.group(0))
            i = m_bar.end()
            continue

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
