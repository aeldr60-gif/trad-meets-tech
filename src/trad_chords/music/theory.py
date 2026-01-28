from __future__ import annotations

"""
Music theory utilities for interpreting ABC notation in a key  and mode aware way.

This module centralizes all pitch class, scale, mode, and chord mapping logic
used throughout the pipeline. It provides consistent conversions between ABC
tokens and pitch classes, derives diatonic and chromatic scale degree bins,
parses TheSession-style key/mode fields, and translates between chord symbols
and Nashville-number labels. These helpers ensure that every stage,from note
tokenization to feature engineering to chord prediction,uses the same
music theoretic rules.

Public API (used across the project):
- split_key_and_mode(mode_str)
- parse_mode(mode_str) -> (tonic_pc, KeyMode)
- scale_pitch_classes(tonic_pc, key_mode)
- abc_note_to_pitch_class(note_token, tonic_pc=None, key_mode=None)
- degree_bin_label(tonic_pc, scale_pcs, pitch_class)
- chord_symbol_to_pitch_class(chord_symbol) -> (root_pc, quality)
- chord_to_nashville(tonic_pc, scale_pcs, root_pc, quality)
- nashville_to_chord_symbol(tonic_pc, key_mode, nash)
- mode_to_abc_key(mode_str)

The function names intentionally mirror common imports to keep call-sites
predictable and uniform across the codebase.
"""


import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# -------------------------
# Basic pitch-class helpers
# -------------------------

LETTER_TO_PC_NATURAL: Dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

PC_TO_NAME_SHARP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

# Common modes (intervals from tonic)
MODE_INTERVALS: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],  # natural minor / aeolian
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}


def _normalize_mode_name(mode: str) -> str:
    m = (mode or "").strip().lower()
    if not m:
        return "major"
    if m in MODE_INTERVALS:
        return m
    # Common variants
    if m.endswith("maj"):
        return "major"
    if m.endswith("min"):
        return "minor"
    if m == "maj":
        return "major"
    if m == "min":
        return "minor"
    return m


# -------------------------
# Key/mode parsing
# -------------------------

# Matches e.g. Dmajor, Edorian, F#minor, Bbmajor
_MODE_RE = re.compile(r"^(?P<key>[A-Ga-g])(?P<accidental>[#b]?)(?P<mode>[A-Za-z]+)?$")


@dataclass(frozen=True)
class KeyMode:
    tonic_pc: int
    tonic_name: str  # e.g. "D", "F#"
    music_mode: str  # e.g. "major", "dorian"

    @property
    def mode_name(self) -> str:
        """Back-compat alias for older code."""
        return self.music_mode

    @property
    def intervals(self) -> List[int]:
        mode = _normalize_mode_name(self.music_mode)
        return MODE_INTERVALS.get(mode, MODE_INTERVALS["major"])

    @property
    def scale_pcs_absolute(self) -> List[int]:
        return [int((self.tonic_pc + i) % 12) for i in self.intervals]

    @property
    def key_signature_pc_by_letter(self) -> Dict[str, int]:
        """Mapping from letter name (A-G) to pitch class implied by key signature.

        Example: D major -> {D:2,E:4,F:6,G:7,A:9,B:11,C:1}
        """
        tonic_letter = self.tonic_name[0].upper()
        letters = ["C", "D", "E", "F", "G", "A", "B"]
        idx = letters.index(tonic_letter)
        rotated = letters[idx:] + letters[:idx]
        pcs = self.scale_pcs_absolute
        return {rotated[i]: pcs[i] for i in range(7)}


def split_key_and_mode(mode_str: str) -> Tuple[str, str]:
    """Split a combined field like "Edorian" into (key="E", music_mode="Dorian")."""
    s = (mode_str or "").strip()
    if not s:
        return "C", "Major"

    m = _MODE_RE.match(s)
    if not m:
        # fallback: assume first char is key
        return s[:1].upper(), s[1:].capitalize() if len(s) > 1 else "Major"

    key = m.group("key").upper() + (m.group("accidental") or "")
    mode = m.group("mode") or "Major"
    # normalize common forms
    mode_norm = mode.strip().capitalize()
    if mode_norm.lower() in ("maj", "major"):
        mode_norm = "Major"
    if mode_norm.lower() in ("min", "minor"):
        mode_norm = "Minor"
    return key, mode_norm


def parse_mode(mode_str: str) -> Tuple[int, KeyMode]:
    """Parse mode field into tonic pitch class and a KeyMode object."""
    key, music_mode = split_key_and_mode(mode_str)

    m = _MODE_RE.match(key + music_mode)  # cheap way to reuse regex
    if not m:
        tonic_letter = key[0].upper()
        accidental = "#" if "#" in key else ("b" if "b" in key else "")
        mode = _normalize_mode_name(music_mode)
    else:
        tonic_letter = m.group("key").upper()
        accidental = m.group("accidental") or ""
        mode = _normalize_mode_name(m.group("mode") or music_mode)

    tonic_pc = LETTER_TO_PC_NATURAL.get(tonic_letter, 0)
    if accidental == "#":
        tonic_pc = (tonic_pc + 1) % 12
    elif accidental == "b":
        tonic_pc = (tonic_pc - 1) % 12

    tonic_name = tonic_letter + accidental
    km = KeyMode(tonic_pc=int(tonic_pc), tonic_name=tonic_name, music_mode=mode)
    return int(tonic_pc), km


def scale_pitch_classes(tonic_pc: int, key_mode: KeyMode) -> List[int]:
    """Return pitch classes in the scale for this key/mode."""
    # tonic_pc arg kept for call-site compatibility; key_mode already includes it
    return key_mode.scale_pcs_absolute


# -------------------------
# Notes -> pitch class (with key signature)
# -------------------------

_ABC_NOTE_RE = re.compile(
    r"(?P<accidental>\^{1,2}|_{1,2}|=)?(?P<note>[A-Ga-g])(?P<octave>[,']*)?(?P<length>\d+|/\d+|/)?"
)


def abc_note_to_pitch_class(note_token: str, tonic_pc: Optional[int] = None, key_mode: Optional[KeyMode] = None) -> int:
    """Convert an ABC note token to pitch class.

    If key_mode is provided and the token has *no* explicit accidental, we apply
    the key signature implied by the tune's K:/mode.

    This is not a full ABC interpreter (it does not model bar-scoped accidentals).
    """
    m = _ABC_NOTE_RE.match(note_token.strip())
    if not m:
        raise ValueError(f"Not an ABC note token: {note_token!r}")

    accidental = m.group("accidental")
    letter = m.group("note").upper()

    natural_pc = LETTER_TO_PC_NATURAL[letter]

    if accidental is None:
        if key_mode is not None:
            # key signature mapping provides the implied pitch class for this letter
            sig_map = key_mode.key_signature_pc_by_letter
            return int(sig_map.get(letter, natural_pc))
        return int(natural_pc)

    if accidental.startswith("="):
        return int(natural_pc)

    pc = natural_pc
    if accidental.startswith("^"):
        pc += accidental.count("^")
    if accidental.startswith("_"):
        pc -= accidental.count("_")

    return int(pc % 12)


# -------------------------
# Pitch class -> degree bin label
# -------------------------

def degree_bin_label(tonic_pc: int, scale_pcs: List[int], pc: int) -> str:
    """Return the degree-bin column name for a pitch class.

    - diatonic notes map to deg_1..deg_7
    - chromatic notes map to deg_X_5 (in-between degrees)

    The mapping is relative to the *scale* for the tune (mode-aware).
    """
    pc = int(pc) % 12
    if not scale_pcs or len(scale_pcs) != 7:
        scale_pcs = MODE_INTERVALS["major"]

    # Ensure ascending scale pcs starting at tonic
    scale_pcs = [int(p) % 12 for p in scale_pcs]

    if pc in scale_pcs:
        deg = scale_pcs.index(pc) + 1
        return f"deg_{deg}"

    # Find the diatonic degree just below this pc (mod 12), then mark as +0.5
    # Work on an unwrapped 0..23 space to handle wrap-around.
    unwrapped = scale_pcs + [p + 12 for p in scale_pcs]
    candidates = []
    for i, p in enumerate(unwrapped[:7]):
        nxt = unwrapped[i + 1]
        lo = p
        hi = nxt
        pc_u = pc if pc >= lo else pc + 12
        if lo < pc_u < hi:
            base_deg = (i % 7) + 1
            return f"deg_{base_deg}_5"

    # Fallback: put in the closest bin by semitone distance
    # (rare edge cases with malformed mode strings)
    closest_i = min(range(7), key=lambda i: min((pc - scale_pcs[i]) % 12, (scale_pcs[i] - pc) % 12))
    return f"deg_{closest_i + 1}_5"


# -------------------------
# Chords
# -------------------------

_CHORD_RE = re.compile(r"^(?P<root>[A-Ga-g])(?P<accidental>[#b]?)(?P<rest>.*)$")


def chord_symbol_to_pitch_class(chord_symbol: str) -> Tuple[int, str]:
    """Parse a chord symbol (from ABC quotes) into (root pitch class, quality).

    Quality is simplified into 'maj' or 'min' for baseline modeling.
    """
    s = (chord_symbol or "").strip()
    if not s:
        raise ValueError("Empty chord symbol")

    m = _CHORD_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized chord symbol: {chord_symbol!r}")

    root_letter = m.group("root").upper()
    acc = (m.group("accidental") or "")
    rest = (m.group("rest") or "").strip().lower()

    root_pc = LETTER_TO_PC_NATURAL[root_letter]
    if acc == "#":
        root_pc = (root_pc + 1) % 12
    elif acc == "b":
        root_pc = (root_pc - 1) % 12

    # Very lightweight quality detection
    quality = "maj"
    # 'm' often indicates minor, but avoid 'maj'
    if rest.startswith("m") and not rest.startswith("maj"):
        quality = "min"
    if "min" in rest:
        quality = "min"

    return int(root_pc), quality


def chord_to_nashville(tonic_pc: int, scale_pcs: List[int], root_pc: int, quality: str) -> str:
    """Convert a chord root/quality into a Nashville label like 'deg_5:maj'."""
    deg = degree_bin_label(int(tonic_pc), scale_pcs, int(root_pc))
    q = "min" if (quality or "").lower().startswith("min") else "maj"
    return f"{deg}:{q}"


def nashville_to_chord_symbol(tonic_pc: int, key_mode: KeyMode, nash: str) -> str:
    """Convert a Nashville label (deg_X:maj/min) back to a simple chord symbol."""
    s = (nash or "").strip()
    if not s:
        return ""

    if ":" in s:
        deg_part, qual = s.split(":", 1)
    else:
        deg_part, qual = s, "maj"

    # deg_1 or deg_1_5
    m = re.match(r"deg_(?P<n>\d)(?P<half>_5)?$", deg_part)
    if not m:
        return ""

    n = int(m.group("n"))
    half = m.group("half") is not None

    scale = key_mode.scale_pcs_absolute
    root_pc = scale[n - 1]
    if half:
        # chromatic in-between degree -> +1 semitone from the lower degree
        root_pc = (root_pc + 1) % 12

    root_name = PC_TO_NAME_SHARP[int(root_pc)]
    if (qual or "").lower().startswith("min"):
        return root_name + "m"
    return root_name


def mode_to_abc_key(mode_str: str) -> str:
    """Turn theSession mode field into an ABC K: value."""
    key, mm = split_key_and_mode(mode_str)
    mm_l = mm.lower()
    if mm_l == "major":
        return f"{key}maj"
    if mm_l == "minor":
        return f"{key}min"
    # ABC uses 'dor', 'mix', etc commonly; keep readable
    return f"{key}{mm_l[:3]}"
