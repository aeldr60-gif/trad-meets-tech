# Trad Meets Tech: Predicting Chords for Jigs and Reels

This WGU capstone project learns chord **placement** and **tone** from Irish trad tunes that include chords, then predicts chords for chordless jigs and reels using TheSession CSV export.

## Data expected (your schema)
`tunes.csv` (tab-separated) must include columns:
- tune_id, name, type, meter, mode, abc

`tune_popularity.csv` should include tune_id + a popularity count column (often `tunebooks`).

Place them here:
- `data/raw/tunes.csv`
- `data/raw/tune_popularity.csv`

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e .
