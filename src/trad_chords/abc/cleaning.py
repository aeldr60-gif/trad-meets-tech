import re

"""
Removes common ABC-notation decorations, such as !markers!, +ornaments+, {grace
notes}, and simple symbols like ~ and . to produce a cleaner string suitable for
analysis. Also collapses repeated whitespace.

Example:
    Input:  "A{g}B !trill! c~ d+cut+ e."
    Output: "A B c d e"
"""


_DECORATION_PATTERNS = [
    r"!\w+!",            # !slide!
    r"\+[^+]+\+",        # +ornament+
    r"\{[^}]*\}",        # {grace notes}
]

# Common inline markers we can safely drop for analysis
_SIMPLE_DROP = [
    "~",  # roll/long note marker
    ".",  # staccato/spacing sometimes used
]


def remove_decorations(abc: str) -> str:
    s = abc
    for pat in _DECORATION_PATTERNS:
        s = re.sub(pat, "", s)
    for ch in _SIMPLE_DROP:
        s = s.replace(ch, "")
    # compress repeated whitespace
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()
