#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # ensure headless operation
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
    return pd.read_csv(path, low_memory=False)


def savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[OK] Wrote {outpath}")


def ensure_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)


def normalize_tune_type(s: str) -> str:
    s = (s or "").strip().lower()
    if "jig" in s:
        return "jig"
    if "reel" in s:
        return "reel"
    return s or "unknown"


def dominant_slots_per_measure(df: pd.DataFrame, type_col: str, spm_col: str, t: str) -> int | None:
    sub = df[df[type_col] == t]
    if sub.empty:
        return None
    vc = sub[spm_col].value_counts(dropna=True)
    if vc.empty:
        return None
    try:
        return int(vc.idxmax())
    except Exception:
        return None


def sorted_degree_labels(labels: list[str]) -> list[str]:
    def key_fn(x: str):
        try:
            return float(str(x))
        except Exception:
            return 999.0
    return sorted(labels, key=key_fn)


# ----------------------------
# Plot 01: Feature set sweep (unchanged)
# ----------------------------
def plot_feature_set_sweep(sweep_csv: Path, outdir: Path) -> None:
    df = read_csv_safe(sweep_csv)
    if df is None or df.empty:
        return

    fs_col = pick_col(df, ["feature_set", "feature_set_name", "name"])
    p_col = pick_col(df, ["placement_accuracy", "placement_acc", "placement"])
    t_col = pick_col(df, ["tone_accuracy", "tone_acc", "tone"])

    if not (fs_col and p_col and t_col):
        print("[WARN] feature_set_sweep missing expected columns; skipping.")
        print("       Have columns:", list(df.columns))
        return

    df = df.copy()
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
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


# ----------------------------
# Plot 02: Chord frequency (top-K) with tune type legend (jig vs reel)
# ----------------------------
def plot_chord_frequency_by_type(beat_csv: Path, outdir: Path, top_k: int = 20) -> None:
    df = read_csv_safe(beat_csv)
    if df is None or df.empty:
        return

    type_col = pick_col(df, ["type"])
    has_col = pick_col(df, ["has_chord_here"])
    chord_col = pick_col(df, ["chord_nashville", "chord_label"])

    if not (type_col and has_col and chord_col):
        print("[WARN] beat_slots missing required columns for chord frequency plot.")
        print("       Need: type, has_chord_here, chord_nashville/chord_label")
        print("       Have:", list(df.columns))
        return

    d = df[df[has_col] == 1].copy()
    d[type_col] = d[type_col].astype(str).map(normalize_tune_type)

    # focus on jig vs reel
    d = d[d[type_col].isin(["jig", "reel"])].copy()
    if d.empty:
        print("[WARN] No jig/reel chord rows found; skipping chord frequency by type.")
        return

    top_labels = d[chord_col].astype(str).value_counts().head(top_k).index.tolist()
    d = d[d[chord_col].astype(str).isin(top_labels)].copy()

    pivot = pd.pivot_table(
        d,
        index=chord_col,
        columns=type_col,
        values=has_col,
        aggfunc="count",
        fill_value=0,
    ).reindex(top_labels)

    types = [c for c in ["jig", "reel"] if c in pivot.columns]
    x = np.arange(len(pivot.index))
    width = 0.4 if len(types) == 2 else 0.6

    fig = plt.figure(figsize=(max(10, 0.55 * len(pivot.index)), 5))
    ax = fig.add_subplot(111)

    if len(types) == 2:
        ax.bar(x - width / 2, pivot[types[0]].values, width, label=types[0])
        ax.bar(x + width / 2, pivot[types[1]].values, width, label=types[1])
    else:
        ax.bar(x, pivot[types[0]].values, width, label=types[0])

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in pivot.index], rotation=45, ha="right")
    ax.set_title(f"Top {top_k} Chords by Tune Type (Jig vs Reel)")
    ax.set_xlabel("Chord label")
    ax.set_ylabel("Count")
    ax.legend(title="Tune type")

    savefig(fig, outdir / "02_chord_frequency_by_type.png")


# ----------------------------
# Plot 03: Chord position within measure (slot_position) with type legend
# ----------------------------
def plot_chord_count_by_slot_and_type(beat_csv: Path, outdir: Path) -> None:
    df = read_csv_safe(beat_csv)
    if df is None or df.empty:
        return

    type_col = pick_col(df, ["type"])
    has_col = pick_col(df, ["has_chord_here"])
    slot_col = pick_col(df, ["slot_position"])
    spm_col = pick_col(df, ["slots_per_measure"])

    if not (type_col and has_col and slot_col and spm_col):
        print("[WARN] beat_slots missing required columns for slot-position chord count plot.")
        return

    d = df[df[has_col] == 1].copy()
    d[type_col] = d[type_col].astype(str).map(normalize_tune_type)
    d = d[d[type_col].isin(["jig", "reel"])].copy()
    if d.empty:
        print("[WARN] No jig/reel chord rows found; skipping chord count by slot plot.")
        return

    # For fairness, restrict each type to its dominant slots_per_measure
    jig_spm = dominant_slots_per_measure(d, type_col, spm_col, "jig")
    reel_spm = dominant_slots_per_measure(d, type_col, spm_col, "reel")

    dd = []
    if jig_spm is not None:
        dd.append(d[(d[type_col] == "jig") & (d[spm_col] == jig_spm)])
    if reel_spm is not None:
        dd.append(d[(d[type_col] == "reel") & (d[spm_col] == reel_spm)])

    if not dd:
        print("[WARN] Could not determine dominant slots_per_measure for jig/reel; skipping.")
        return

    d2 = pd.concat(dd, ignore_index=True)

    # Build counts for x-axis up to max spm across both
    max_spm = int(d2[spm_col].max())
    x_slots = np.arange(1, max_spm + 1)

    counts = (
        d2.groupby([type_col, slot_col])
          .size()
          .unstack(type_col, fill_value=0)
          .reindex(x_slots, fill_value=0)
    )

    types = [t for t in ["jig", "reel"] if t in counts.columns]
    x = np.arange(len(x_slots))
    width = 0.4 if len(types) == 2 else 0.6

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    if len(types) == 2:
        ax.bar(x - width / 2, counts[types[0]].values, width, label=f"{types[0]} (spm≈{jig_spm})")
        ax.bar(x + width / 2, counts[types[1]].values, width, label=f"{types[1]} (spm≈{reel_spm})")
    else:
        ax.bar(x, counts[types[0]].values, width, label=types[0])

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in x_slots])
    ax.set_title("Chord Count by Slot Position (Jig vs Reel)")
    ax.set_xlabel("slot_position within measure")
    ax.set_ylabel("Chord count (has_chord_here = 1)")
    ax.legend()

    savefig(fig, outdir / "03_chord_count_by_slot_and_type.png")


# ----------------------------
# Plot 04/05: Heatmaps of chord root degree frequency by slot position
#   - one for jigs, one for reels
# ----------------------------
def plot_chord_root_degree_heatmap_by_type(beat_csv: Path, outdir: Path, tune_type: str, outname: str) -> None:
    df = read_csv_safe(beat_csv)
    if df is None or df.empty:
        return

    type_col = pick_col(df, ["type"])
    has_col = pick_col(df, ["has_chord_here"])
    slot_col = pick_col(df, ["slot_position"])
    spm_col = pick_col(df, ["slots_per_measure"])
    deg_col = pick_col(df, ["chord_root_degree"])

    if not (type_col and has_col and slot_col and spm_col and deg_col):
        print("[WARN] beat_slots missing required columns for chord root-degree heatmap.")
        return

    d = df[df[has_col] == 1].copy()
    d[type_col] = d[type_col].astype(str).map(normalize_tune_type)
    d = d[d[type_col] == tune_type].copy()
    if d.empty:
        print(f"[WARN] No chord rows for type={tune_type}; skipping heatmap.")
        return

    dom_spm = dominant_slots_per_measure(d, type_col, spm_col, tune_type)
    if dom_spm is None:
        print(f"[WARN] Could not determine dominant slots_per_measure for {tune_type}; skipping heatmap.")
        return

    d = d[d[spm_col] == dom_spm].copy()
    d[slot_col] = pd.to_numeric(d[slot_col], errors="coerce")
    d = d.dropna(subset=[slot_col])
    d[slot_col] = d[slot_col].astype(int)

    d[deg_col] = d[deg_col].astype(str)
    degree_labels = sorted_degree_labels([x for x in d[deg_col].unique() if x != "nan"])

    # Pivot counts: rows = degree, cols = slot_position
    pivot = pd.pivot_table(
        d,
        index=deg_col,
        columns=slot_col,
        values=has_col,
        aggfunc="count",
        fill_value=0,
    ).reindex(index=degree_labels)

    # Ensure all slot positions exist
    pivot = pivot.reindex(columns=list(range(1, dom_spm + 1)), fill_value=0)

    fig = plt.figure(figsize=(10, max(5, 0.35 * len(pivot.index))))
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(f"Chord Root Degree Frequency by Slot ({tune_type.title()}, slots_per_measure≈{dom_spm})")
    ax.set_xlabel("slot_position")
    ax.set_ylabel("chord_root_degree")

    ax.set_xticks(np.arange(dom_spm))
    ax.set_xticklabels([str(i) for i in range(1, dom_spm + 1)])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    savefig(fig, outdir / outname)


# ----------------------------
# Plot 06: chord_quality distribution by tune type (stacked proportions)
# ----------------------------
def plot_chord_quality_by_type(beat_csv: Path, outdir: Path, top_k: int = 8) -> None:
    df = read_csv_safe(beat_csv)
    if df is None or df.empty:
        return

    type_col = pick_col(df, ["type"])
    has_col = pick_col(df, ["has_chord_here"])
    qual_col = pick_col(df, ["chord_quality"])

    if not (type_col and has_col and qual_col):
        print("[WARN] beat_slots missing required columns for chord_quality plot.")
        return

    d = df[df[has_col] == 1].copy()
    d[type_col] = d[type_col].astype(str).map(normalize_tune_type)
    d = d[d[type_col].isin(["jig", "reel"])].copy()
    if d.empty:
        print("[WARN] No jig/reel chord rows found; skipping chord_quality plot.")
        return

    # Keep top qualities overall, bucket rest
    top_quals = d[qual_col].astype(str).value_counts().head(top_k).index.tolist()
    d["qual_b"] = np.where(d[qual_col].astype(str).isin(top_quals), d[qual_col].astype(str), "OTHER")

    pivot = pd.pivot_table(
        d,
        index=type_col,
        columns="qual_b",
        values=has_col,
        aggfunc="count",
        fill_value=0,
    ).reindex(index=["jig", "reel"])

    # Convert to proportions for stacked bars
    props = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    bottom = np.zeros(len(props.index))
    x = np.arange(len(props.index))

    for col in props.columns:
        ax.bar(x, props[col].values, bottom=bottom, label=str(col))
        bottom += props[col].values

    ax.set_xticks(x)
    ax.set_xticklabels([t.title() for t in props.index])
    ax.set_ylim(0, 1.0)
    ax.set_title("Chord Quality Distribution by Tune Type (Proportions)")
    ax.set_ylabel("Proportion of chords")
    ax.legend(title="chord_quality", bbox_to_anchor=(1.02, 1), loc="upper left")

    savefig(fig, outdir / "06_chord_quality_by_type.png")





# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ensure_repo_root()
    warnings.filterwarnings("ignore", category=FutureWarning)

    outdir = Path("report_figures")
    outdir.mkdir(parents=True, exist_ok=True)

    sweep = Path("outputs/evaluation/feature_set_sweep.csv")
    beat = Path("data/processed/beat_slots_topn.csv")

    print("[INFO] CWD:", Path.cwd())
    print("[INFO] Inputs:",
          "feature_set_sweep=", sweep.exists(),
          "beat_slots_topn=", beat.exists())
    print("[INFO] Output dir:", outdir.resolve())

    plot_feature_set_sweep(sweep, outdir)
    plot_chord_frequency_by_type(beat, outdir, top_k=20)
    plot_chord_count_by_slot_and_type(beat, outdir)

    # Heatmaps (split: jigs vs reels)
    plot_chord_root_degree_heatmap_by_type(beat, outdir, tune_type="jig", outname="04_chord_root_degree_heatmap_jigs.png")
    plot_chord_root_degree_heatmap_by_type(beat, outdir, tune_type="reel", outname="05_chord_root_degree_heatmap_reels.png")

    # Optional extras you asked for
    plot_chord_quality_by_type(beat, outdir, top_k=8)
    plot_melody_degree_chord_vs_nochord(beat, outdir)

    print("\nDone. Figures are in:", outdir.resolve())


if __name__ == "__main__":
    main()
