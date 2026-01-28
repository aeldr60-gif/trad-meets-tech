# Trad Meets Tech (trad-chords)

Predicting chord placement and chord identity for Irish traditional jigs and reels using TheSession.org ABC notation.

## What this project does
This pipeline:
1) downloads TheSession CSV data,
2) filters to jigs and reels,
3) tokenizes ABC into a notes table with part/measure structure,
4) converts notes into beat-aligned “slot” features,
5) trains two baseline ML models:
   - chord placement (does a chord occur at this slot?)
   - chord tone (which chord label, in Nashville-style notation, at true chord slots?)
6) applies the models to chordless tunes and outputs ABC strings with inserted chord annotations.

## Quickstart

### 1) Setup
```bash
cd ~/trad-meets-tech
conda deactivate || true
source .venv/bin/activate
pip install -e .

### 2) Running the pipeline

trad-chords fetch-data
trad-chords load-data
trad-chords build-index
trad-chords build-notes-table
trad-chords build-beat-slots
trad-chords split-index
trad-chords train
trad-chords evaluate-selfcheck
trad-chords sweep-feature-sets
trad-chords harmonize-chordless