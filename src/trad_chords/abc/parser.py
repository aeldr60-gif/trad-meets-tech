from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


BAR_TOKENS = {"|", "||", "|:", ":|", "::", "[|", "|]"}
ENDING_START_RE = re.compile(r"\[\d")  # [1 [2
CHORD_RE = re.compile(r'"{1,2}([^"]+?)"{1,2}')



@dataclass(frozen=True)
class AbcToken:
    kind: str  # note, rest, bar, chord, other
    text: str


_NOTE_RE = re.compile(
    r"""
    (?P<accidental>\^{1,2}|_{1,2}|=)?   # ^ ^^ _ __ =
    (?P<note>[A-Ga-g])                 # letter
    (?P<octave>[,']*)                  # octave marks
    (?P<length>\d+|/\d+|/)?            # simple length forms: 2, /, /2
    """,
    re.VERBOSE,
)

_REST_RE = re.compile(r"(?P<rest>[zZxX])(?P<length>\d+|/\d+|/)?")

def tokenize_abc(abc: str) -> Iterator[AbcToken]:
    i = 0
    n = len(abc)

    while i < n:
        # skip whitespace/newlines
        if abc[i].isspace():
            i += 1
            continue

        # chord tokens like ""Am""
        m = CHORD_RE.match(abc, i)
        if m:
            yield AbcToken("chord", m.group(1).strip())
            i = m.end()
            continue

        # endings like [1 [2
        if ENDING_START_RE.match(abc, i):
            yield AbcToken("ending", abc[i:i+2])
            i += 2
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
