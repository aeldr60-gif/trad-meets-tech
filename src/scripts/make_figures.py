#!/usr/bin/env python3
"""
Generate report-ready matplotlib figures for Trad Meets Tech.

Inputs (defaults):
  - outputs/evaluation/feature_set_sweep.csv
  - outputs/evaluation/selfcheck_summary.csv
  - data/processed/beat_slots_topn.csv
  - data/processed/notes_table_topn.csv

Outputs:
  - report_figures/*.png

Run:
  python scripts/make_figures.py
  python scripts/make_figures.py --outdir report_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    # low_memory=False avoids mixed dtype chunk inference warnings
    return pd.read_csv(path, low_memory=False)


def savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {outpath}")


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ----------------------------
# Plots
# ----------------------------
def plot_feature_set_sweep(sweep_csv: Path, outdir: Path) -> None:
    df = read_csv_safe(sweep_csv)
    if df is None or df.empty:
        return

    # Try to normalize expected column names
    fs_col = pick_col(df, ["feature_set", "feature_set_name", "name"])
    p_col = pick_col(df, ["placement_accuracy", "placement_acc", "placement"])
    t_col = pick_col(df, ["tone_accuracy", "tone_acc", "tone"])

    if not (fs_col and p_col and t_col):
        print("[WARN] feature_set_sweep missing expected columns; skipping sweep plot.")
        print("       Have columns:", list(df.columns))
        return

    df = df.copy()
    df[p_col] = safe_to_numeric(df[p_col])
    df[t_col] = safe_to_numeric(df[t_col])
    df = df.sort_values([t_col, p_col], ascending=False)

    x = np.arange(len(df))
    width = 0.4

    fig = plt.figure(figsize=(max(8, 1.3 * len(df)), 5))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, df[p_col].values, width, label="Placement accuracy")
    ax.bar(x + width / 2, df[t_col].values, width, label="Tone accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(df[fs_col].astype(str).tolist(), rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Feature Set Sweep: Placement vs Tone Accuracy")
    ax.set_ylabel("Accuracy")
    ax.legend()

    savefig(fig, outdir / "01_feature_set_sweep.png")


def plot_tone_confusion(selfcheck_csv: Path, outdir: Path, top_k: int = 15) -> None:
    df = read_csv_safe(selfcheck_csv)
    if df is None or df.empty:
        return

    # Common column name patterns (robust to minor renames)
    true_tone = pick_col(df, ["true_chord_nashville", "y_true_tone", "tone_true", "chord_true", "true_tone"])
    pred_tone = pick_col(df, ["pred_chord_nashville", "y_pred_tone", "tone_pred", "chord_pred", "pred_tone"])

    # If eval only stored tone on chord slots, fine. Otherwise we filter to rows with non-null true tone.
    if not (true_tone and pred_tone):
        print("[WARN] selfcheck_summary missing tone true/pred columns; skipping confusion matrix.")
        print("       Have columns:", list(df.columns))
        return

    d = df[[true_tone, pred_tone]].copy()
    d[true_tone] = d[true_tone].astype(str)
    d[pred_tone] = d[pred_tone].astype(str)

    # Drop empty-ish rows
    d = d[(d[true_tone].notna()) & (d[true_tone] != "nan") & (d[true_tone].str.len() > 0)]
    if d.empty:
        print("[WARN] No tone rows available after filtering; skipping confusion matrix.")
        return

    # Top-K classes by true frequency; bucket the rest into OTHER
    top_labels = d[true_tone].value_counts().head(top_k).index.tolist()
    d["true_b"] = np.where(d[true_tone].isin(top_labels), d[true_tone], "OTHER")
    d["pred_b"] = np.where(d[pred_tone].isin(top_labels), d[pred_tone], "OTHER")

    cm = pd.crosstab(d["true_b"], d["pred_b"])
    # Ensure consistent ordering (top_labels + OTHER)
    labels = top_labels + (["OTHER"] if "OTHER" in cm.index.union(cm.columns) else [])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm.values, aspect="auto")

    ax.set_title(f"Tone Confusion Matrix (Top {top_k} Nashville Labels + OTHER)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Light annotation only if matrix isn't too large
    if len(labels) <= 16:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = int(cm.values[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, outdir / "02_tone_confusion_matrix.png")


def plot_placement_by_slot(selfcheck_csv: Path, outdir: Path) -> None:
    df = read_csv_safe(selfcheck_csv)
    if df is None or df.empty:
        return

    true_place = pick_col(df, ["true_has_chord_here", "y_true_place", "placement_true", "has_chord_here_true", "has_chord_here"])
    pred_place = pick_col(df, ["pred_has_chord_here", "y_pred_place", "placement_pred", "has_chord_here_pred", "pred_has_chord_here"])
    slot_pos = pick_col(df, ["slot_position", "slot_pos"])
    spm = pick_col(df, ["slots_per_measure", "slots_in_measure", "spm"])

    if not (true_place and pred_place and slot_pos and spm):
        print("[WARN] selfcheck_summary missing placement/slot columns; skipping placement-by-slot plot.")
        print("       Have columns:", list(df.columns))
        return

    d = df[[true_place, pred_place, slot_pos, spm]].copy()
    d[true_place] = safe_to_numeric(d[true_place])
    d[pred_place] = safe_to_numeric(d[pred_place])
    d[slot_pos] = safe_to_numeric(d[slot_pos])
    d[spm] = safe_to_numeric(d[spm])
    d = d.dropna()

    if d.empty:
        print("[WARN] No placement rows available after numeric filtering; skipping placement-by-slot plot.")
        return

    # Focus on the most common slots_per_measure (often 8 for reels, 6 for jigs, etc.)
    most_common_spm = int(d[spm].value_counts().idxmax())
    dd = d[d[spm] == most_common_spm].copy()
    if dd.empty:
        return

    # Precision and recall per slot_position
    rows = []
    for s, g in dd.groupby(slot_pos):
        y_true = g[true_place].astype(int).values
        y_pred = g[pred_place].astype(int).values

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        rows.append((int(s), precision, recall, len(g)))

    m = pd.DataFrame(rows, columns=["slot_position", "precision", "recall", "n"])
    m = m.sort_values("slot_position")

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(m["slot_position"], m["precision"], marker="o", label="Precision")
    ax.plot(m["slot_position"], m["recall"], marker="o", label="Recall")
    ax.set_title(f"Placement Performance by Slot (slots_per_measure={most_common_spm})")
    ax.set_xlabel("Slot position within measure")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(m["slot_position"].tolist())
    ax.legend()

    savefig(fig, outdir / "03_placement_precision_recall_by_slot.png")


def plot_chord_frequency(beat_slots_csv: Path, outdir: Path, top_k: int = 20) -> None:
    df = read_csv_safe(beat_slots_csv)
    if df is None or df.empty:
        return

    has_chord = pick_col(df, ["has_chord_here"])
    chord = pick_col(df, ["chord_nashville", "chord_label", "nashville"])
    if not (has_chord and chord):
        print("[WARN] beat_slots_topn missing has_chord_here/chord_nashville; skipping chord frequency plot.")
        print("       Have columns:", list(df.columns))
        return

    d = df[df[has_chord] == 1].copy()
    if d.empty:
        print("[WARN] No chord slots found in beat_slots_topn; skipping chord frequency plot.")
        return

    vc = d[chord].astype(str).value_counts().head(top_k)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(vc.index.tolist(), vc.values)
    ax.set_title(f"Most Common Nashville Chords (Top {top_k})")
    ax.set_xlabel("Chord label")
    ax.set_ylabel("Count")
    ax.set_xticklabels(vc.index.tolist(), rotation=45, ha="right")

    savefig(fig, outdir / "04_chord_frequency_topk.png")


def plot_harmonic_rhythm(beat_slots_csv: Path, outdir: Path) -> None:
    df = read_csv_safe(beat_slots_csv)
    if df is None or df.empty:
        return

    need = ["setting_id", "part", "measure_number", "slot_position", "has_chord_here", "chord_nashville"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[WARN] beat_slots_topn missing {missing}; skipping harmonic rhythm plot.")
        return

    d = df[df["has_chord_here"] == 1].copy()
    if d.empty:
        print("[WARN] No chord slots; skipping harmonic rhythm plot.")
        return

    # Count chord changes per measure (within each setting_id/part/measure_number)
    changes = []
    for (_, _, _), g in d.groupby(["setting_id", "part", "measure_number"], sort=False):
        g = g.sort_values("slot_position")
        seq = g["chord_nashville"].astype(str).tolist()
        # chord changes = transitions where label differs from previous
        n_changes = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                n_changes += 1
        changes.append(n_changes)

    changes = np.array(changes, dtype=int)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    bins = np.arange(changes.min(), changes.max() + 2) - 0.5
    ax.hist(changes, bins=bins)
    ax.set_title("Harmonic Rhythm: Chord Changes per Measure (Chordy Tunes)")
    ax.set_xlabel("Chord changes per measure")
    ax.set_ylabel("Number of measures")
    ax.set_xticks(np.arange(changes.min(), changes.max() + 1))

    savefig(fig, outdir / "05_harmonic_rhythm_changes_per_measure.png")


def plot_measure_stats(notes_table_csv: Path, outdir: Path) -> None:
    df = read_csv_safe(notes_table_csv)
    if df is None or df.empty:
        return

    need = ["setting_id", "measure_number", "token_kind"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[WARN] notes_table_topn missing {missing}; skipping measure stats plot.")
        return

    # Distribution of number of measures per setting
    max_meas = df.groupby("setting_id")["measure_number"].max()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.hist(max_meas.values, bins=30)
    ax.set_title("Distribution: Measures per Setting")
    ax.set_xlabel("Max measure_number per setting")
    ax.set_ylabel("Count of settings")
    savefig(fig, outdir / "06_measures_per_setting.png")

    # Token kind composition
    vc = df["token_kind"].value_counts().head(12)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(vc.index.astype(str).tolist(), vc.values)
    ax.set_title("Token Kind Counts (Top 12) in notes_table_topn")
    ax.set_xlabel("token_kind")
    ax.set_ylabel("Count")
    ax.set_xticklabels(vc.index.astype(str).tolist(), rotation=30, ha="right")
    savefig(fig, outdir / "07_token_kind_counts.png")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="outputs/evaluation/feature_set_sweep.csv")
    parser.add_argument("--selfcheck", type=str, default="outputs/evaluation/selfcheck_summary.csv")
    parser.add_argument("--beat-slots", type=str, default="data/processed/beat_slots_topn.csv")
    parser.add_argument("--notes-table", type=str, default="data/processed/notes_table_topn.csv")
    parser.add_argument("--outdir", type=str, default="report_figures")
    parser.add_argument("--topk-tone", type=int, default=15)
    parser.add_argument("--topk-chords", type=int, default=20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reduce noisy warnings in console output
    warnings.filterwarnings("ignore", category=FutureWarning)

    plot_feature_set_sweep(Path(args.sweep), outdir)
    plot_tone_confusion(Path(args.selfcheck), outdir, top_k=args.topk-tone)
    plot_placement_by_slot(Path(args.selfcheck), outdir)
    plot_chord_frequency(Path(args.beat_slots), outdir, top_k=args.topk-chords)
    plot_harmonic_rhythm(Path(args.beat_slots), outdir)
    plot_measure_stats(Path(args.notes_table), outdir)

    print("\nDone. Figures are in:", outdir.resolve())


if __name__ == "__main__":
    main()
