from __future__ import annotations

"""Music-theory helpers used by feature builders.

This module is intentionally lightweight (no external deps) and focused on:

1) Parsing TheSession "mode" strings like:
   - "Edorian"  -> tonic="E", music_mode="dorian"
   - "Gmajor"   -> tonic="G", music_mode="ionian" (aka major)
   - "Bminor"   -> tonic="B", music_mode="aeolian" (aka natural minor)
   - "Amixolydian" -> tonic="A", music_mode="mixolydian"

2) Converting ABC note tokens (e.g., "^c", "_B", "E2", "c'", "A,")
   to pitch class (0-11) in C-based semitones.

3) Mapping pitch classes into *key-agnostic* scale-degree bins.

We represent chromatic (non-diatonic) notes as "in-between" degrees.
For example, in D major:
  - C# is deg_7
  - C natural is deg_6_5 (between deg_6 (B) and deg_7 (C#))

This preserves chromatic info without discarding it.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Pitch-class utilities
# -----------------------------

NOTE_TO_PC: Dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


_ABC_NOTE_RE = re.compile(r"^(?P<acc>\^\^|\^|__|_|=)?(?P<note>[A-Ga-g])")


def abc_note_to_pitch_class(token_text: str) -> Optional[int]:
    """Return pitch class (0-11) for an ABC note token, or None if not a note.

    We intentionally ignore octave (', and ') and duration digits.
    """
    if not token_text:
        return None

    m = _ABC_NOTE_RE.match(token_text.strip())
    if not m:
        return None

    acc = m.group("acc") or ""
    note = m.group("note").upper()
    if note not in NOTE_TO_PC:
        return None

    pc = NOTE_TO_PC[note]
    if acc == "^":
        pc += 1
    elif acc == "^^":
        pc += 2
    elif acc == "_":
        pc -= 1
    elif acc == "__":
        pc -= 2
    elif acc == "=":
        pc += 0

    return pc % 12


# -----------------------------
# Mode / key parsing
# -----------------------------

MODE_INTERVALS: Dict[str, List[int]] = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}


@dataclass(frozen=True)
class KeyMode:
    tonic_name: str         # e.g., "D", "F#", "Eb"
    tonic_pc: int           # 0-11
    music_mode: str         # normalized, e.g., "dorian", "ionian", "aeolian"

    @property
    def scale_pcs_relative(self) -> List[int]:
        return MODE_INTERVALS[self.music_mode]

    @property
    def scale_pcs_absolute(self) -> List[int]:
        return [ (self.tonic_pc + x) % 12 for x in self.scale_pcs_relative ]


_MODE_RE = re.compile(r"^(?P<tonic>[A-Ga-g])(?P<acc>[#b]?)(?P<mode>[A-Za-z]+)$")


def parse_mode_string(mode_str: str) -> Optional[KeyMode]:
    """Parse TheSession mode strings like "Edorian" or "Gmajor".

    Returns KeyMode or None if unparseable.
    """
    if not mode_str or not isinstance(mode_str, str):
        return None

    s = mode_str.strip()
    m = _MODE_RE.match(s)
    if not m:
        return None

    tonic = m.group("tonic").upper()
    acc = m.group("acc")
    mode_raw = m.group("mode").lower()

    # Normalize common names
    if mode_raw == "major":
        mode = "ionian"
    elif mode_raw == "minor":
        mode = "aeolian"
    else:
        mode = mode_raw

    if mode not in MODE_INTERVALS:
        return None

    tonic_pc = NOTE_TO_PC.get(tonic)
    if tonic_pc is None:
        return None

    if acc == "#":
        tonic_pc = (tonic_pc + 1) % 12
        tonic_name = f"{tonic}#"
    elif acc == "b":
        tonic_pc = (tonic_pc - 1) % 12
        tonic_name = f"{tonic}b"
    else:
        tonic_name = tonic

    return KeyMode(tonic_name=tonic_name, tonic_pc=tonic_pc, music_mode=mode)


def split_key_and_mode(mode_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Convenience splitter: "Edorian" -> ("E", "dorian")."""
    km = parse_mode_string(mode_str)
    if not km:
        return None, None
    return km.tonic_name, km.music_mode


# -----------------------------
# Scale-degree binning
# -----------------------------


def pitch_class_to_degree_bin(pc: int, key_mode: KeyMode) -> str:
    """Map pitch class to a degree bin label.

    Returns one of:
      deg_1 .. deg_7 (diatonic)
      deg_1_5 .. deg_7_5 (chromatic between degrees)

    The chromatic bins are defined *ascending* between diatonic degrees.
    """
    rel = (pc - key_mode.tonic_pc) % 12
    degrees = key_mode.scale_pcs_relative  # ascending within an octave

    # Exact diatonic match
    for i, deg_pc in enumerate(degrees, start=1):
        if rel == deg_pc:
            return f"deg_{i}"

    # Find insertion position in circular scale
    # Use last degree <= rel as the "lower" boundary.
    lower_idx = None
    for i, deg_pc in enumerate(degrees, start=1):
        if deg_pc <= rel:
            lower_idx = i
        else:
            break

    if lower_idx is None:
        # rel is below deg_1 but we treat it as between deg_7 and deg_1
        lower_idx = 7

    return f"deg_{lower_idx}_5"


_CHORD_RE = re.compile(r"^(?P<root>[A-Ga-g])(?P<acc>[#b]?)(?P<rest>.*)$")


def parse_chord_symbol(chord_text: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse a simple chord symbol like 'Am', 'G', 'F#min', 'Bb', 'Edim'.

    Returns (root_pitch_class, quality) where quality in {'maj','min','dim','unk'}.
    """
    if not chord_text or not isinstance(chord_text, str):
        return None, None

    s = chord_text.strip()
    m = _CHORD_RE.match(s)
    if not m:
        return None, None

    root = m.group("root").upper()
    acc = m.group("acc")
    rest = (m.group("rest") or "").lower()

    base = NOTE_TO_PC.get(root)
    if base is None:
        return None, None
    if acc == "#":
        base = (base + 1) % 12
    elif acc == "b":
        base = (base - 1) % 12

    # Very lightweight quality detection
    if rest.startswith("m") and not rest.startswith("maj"):
        quality = "min"
    elif "dim" in rest or rest.startswith("o"):
        quality = "dim"
    else:
        quality = "maj"

    return base, quality


def chord_to_nashville(chord_text: str, key_mode: KeyMode) -> Tuple[Optional[str], Optional[str]]:
    """Convert chord text into a key-agnostic Nashville-ish label.

    Example outputs:
      - ("1", "maj")
      - ("6", "min")
      - ("2_5", "maj") for chromatic root
    """
    root_pc, quality = parse_chord_symbol(chord_text)
    if root_pc is None or quality is None:
        return None, None

    deg_bin = pitch_class_to_degree_bin(root_pc, key_mode)
    # deg_3 or deg_3_5 -> "3" or "3_5"
    deg_short = deg_bin.replace("deg_", "")
    return deg_short, quality
