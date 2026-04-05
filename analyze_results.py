"""
Generate thesis figures 2–4 from standardized test CSV + evolution result files.

Figure 1 (standardised test bar chart) is produced separately by:
    plot_standardized_test_evolution_match.py
(do not duplicate that logic here.)

Outputs under results/analysis/:
    figure_2_train_vs_test_scatter.png
    figure_3_8x8_test_win_rate_by_condition.png
    figure_4_fitness_over_generations.png  (8×8 runs only; non-minimax evolution_8x8_*.txt)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# 8x8 experimental condition per result file (see docs/run_tracker.md)
CONDITION_BY_FILE: dict[str, str] = {
    "evolution_8x8_20251216_222007.txt": "Config_5",
    "evolution_8x8_20251218_002100.txt": "Baseline (Config_6)",
    "evolution_8x8_20260216_052949.txt": "Baseline (Config_6)",
    "evolution_8x8_20260218_003143.txt": "Baseline (Config_6)",
    "evolution_8x8_20260219_203646.txt": "Baseline (Config_6)",
    "evolution_8x8_20260223_045614.txt": "Condition A (30% co-ev.)",
    "evolution_8x8_20260225_114105.txt": "Condition A (30% co-ev.)",
    "evolution_8x8_20260227_011809.txt": "Condition A (30% co-ev.)",
    "evolution_8x8_20260302_115155.txt": "Condition B (50% co-ev.)",
    "evolution_8x8_20260307_093010.txt": "Condition B (50% co-ev.)",
    "evolution_8x8_20260309_155326.txt": "Condition B (50% co-ev.)",
    "evolution_8x8_20260311_043951.txt": "Condition C (mut. 0.10)",
    "evolution_8x8_20260312_214315.txt": "Condition C (mut. 0.10)",
    "evolution_8x8_20260316_010459.txt": "Condition C (mut. 0.10)",
    "evolution_8x8_20260318_051106.txt": "Condition D (mut. 0.20)",
    "evolution_8x8_20260320_104123.txt": "Condition D (mut. 0.20)",
    "evolution_8x8_20260322_094426.txt": "Condition D (mut. 0.20)",
}

# Order for figure 3 (Config_5 excluded — early pilot, not part of A–D comparison)
CONDITION_ORDER_FIG3: list[str] = [
    "Baseline (Config_6)",
    "Condition A (30% co-ev.)",
    "Condition B (50% co-ev.)",
    "Condition C (mut. 0.10)",
    "Condition D (mut. 0.20)",
]


def _find_latest_standardized_csv() -> Path | None:
    files = list(RESULTS_DIR.glob("standardized_test_results_evolution_match_*.csv"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _is_minimax_row(row: dict[str, str]) -> bool:
    fn = (row.get("Result File") or "").lower()
    return "minimax" in fn


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def parse_generation_stats(result_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (generations, best_fitness, avg_fitness) from an evolution result .txt file."""
    text = result_path.read_text(encoding="utf-8", errors="replace")
    gens: list[int] = []
    best: list[float] = []
    avg: list[float] = []
    for line in text.splitlines():
        if line.startswith("Generation ") and "Best=" in line:
            m = re.search(r"Generation (\d+): Best=([\d.]+), Avg=([\d.]+)", line)
            if m:
                gens.append(int(m.group(1)))
                best.append(float(m.group(2)))
                avg.append(float(m.group(3)))
    return gens, best, avg


def plot_figure2_train_vs_test(rows: list[dict[str, str]], source_name: str, out_path: Path) -> None:
    train_x: list[float] = []
    test_y: list[float] = []
    colors: list[str] = []
    for row in rows:
        if _is_minimax_row(row):
            continue
        tr = _float(row, "Train Win Rate")
        te = _float(row, "Test Win Rate")
        if np.isnan(tr) or np.isnan(te):
            continue
        train_x.append(tr * 100.0)
        test_y.append(te * 100.0)
        bs = row.get("Board Size", "")
        colors.append("#3498db" if bs == "6x6" else "#27ae60")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(train_x, test_y, c=colors, edgecolors="black", linewidths=0.4, s=45, zorder=3)
    lo, hi = 0.0, 100.0
    (diag,) = ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x (no gap)")
    ax.set_xlabel("Training win rate (%)")
    ax.set_ylabel("Standardised test win rate (%)")
    ax.set_title("Training vs standardised test performance\n" + source_name, fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            diag,
            Patch(facecolor="#3498db", edgecolor="black", label="6×6"),
            Patch(facecolor="#27ae60", edgecolor="black", label="8×8"),
        ],
        loc="lower right",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure3_conditions(rows: list[dict[str, str]], source_name: str, out_path: Path) -> None:
    """Box + jittered points: standardised test win rate (%) per mapped 8×8 run, by condition."""
    by_cond: dict[str, list[float]] = {c: [] for c in CONDITION_ORDER_FIG3}
    for row in rows:
        if _is_minimax_row(row):
            continue
        if row.get("Board Size") != "8x8":
            continue
        fn = row.get("Result File", "")
        label = CONDITION_BY_FILE.get(fn)
        if label is None or label == "Config_5":
            continue
        if label not in by_cond:
            continue
        te = _float(row, "Test Win Rate")
        if not np.isnan(te):
            by_cond[label].append(te * 100.0)

    data = [by_cond[c] for c in CONDITION_ORDER_FIG3 if by_cond[c]]
    labels = [c for c in CONDITION_ORDER_FIG3 if by_cond[c]]
    if not data:
        raise SystemExit("No 8×8 rows matched CONDITION_BY_FILE; check CSV filenames.")

    fig, ax = plt.subplots(figsize=(11, 6.8))
    positions = np.arange(1, len(labels) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
        medianprops=dict(color="#2ca02c", linewidth=2.2),
        whiskerprops=dict(color="black", linewidth=1.25, linestyle="-"),
        capprops=dict(color="black", linewidth=1.25),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#aed6f1")
        patch.set_alpha(0.85)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        x = rng.uniform(positions[i] - 0.12, positions[i] + 0.12, size=len(vals))
        ax.scatter(x, vals, color="#1b4f72", s=28, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Standardised test win rate (%)")
    ax.set_title("8×8: standardised test performance by condition\n" + source_name, fontsize=11)
    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="50% (random baseline)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    legend_elems = [
        Patch(
            facecolor="#aed6f1",
            edgecolor="#2e86ab",
            linewidth=1,
            label="Shaded box: Q1–Q3 (middle 50% of runs in the group)",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#2ca02c",
            linewidth=2.5,
            solid_capstyle="butt",
            label="Horizontal line inside box: median test win rate",
        ),
        Line2D(
            [0.5, 0.5],
            [0, 1],
            color="black",
            linewidth=1.25,
            label="Vertical black lines: whiskers (extend to min / max run)",
        ),
        Line2D(
            [0.35, 0.65],
            [1, 1],
            color="black",
            linewidth=1.25,
            label="Short horizontal caps: ends of each whisker",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1b4f72",
            markeredgecolor="white",
            markersize=9,
            label="Point: one run (that strategy’s standardised test win rate)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Diamond: mean win rate across runs in the group",
        ),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Grey dashed: 50% vs random"),
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.97,
        title="How to read this plot",
    )

    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.30)
    fig.text(
        0.5,
        0.08,
        "Each column is one experimental condition (Chapter 5). "
        "Standardised test: 5 seeds × 30 games vs random (reevaluate_strategies_match_evolution.py).",
        ha="center",
        fontsize=8,
        style="italic",
        color="#333333",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def discover_evolution_8x8_files() -> list[Path]:
    """8×8 evolution result files only (excludes minimax)."""
    files = sorted(RESULTS_DIR.glob("evolution_8x8_*.txt"))
    return [p for p in files if "minimax" not in p.name.lower()]


def plot_figure4_fitness_curves(paths: list[Path], out_path: Path) -> None:
    """One line per run: best fitness vs generation (lower is better)."""
    if not paths:
        raise SystemExit("No evolution result files found for figure 4.")

    series: list[tuple[Path, list[int], list[float]]] = []
    for path in paths:
        gens, best, _avg = parse_generation_stats(path)
        if not gens:
            print(f"Warning: no generation stats in {path.name}, skipping")
            continue
        series.append((path, gens, best))

    if not series:
        raise SystemExit("No files contained Generation statistics lines.")

    try:
        cmap = plt.cm.turbo
    except AttributeError:
        cmap = plt.cm.viridis
    n = len(series)
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (_path, gens, best) in enumerate(series):
        color = cmap(0.08 + 0.84 * (i / max(n - 1, 1)))
        ax.plot(gens, best, color=color, linewidth=1.05, alpha=0.88)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness in population (lower is better)")
    ax.invert_yaxis()
    ax.set_title(f"8×8: best fitness over generations ({n} runs, alphabetical order)")
    ax.grid(True, alpha=0.35)

    fig.subplots_adjust(bottom=0.14)
    fig.text(
        0.5,
        0.02,
        "Each line: one 8×8 run’s best fitness during training (lower is better).",
        ha="center",
        fontsize=8,
        style="italic",
        color="#333333",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EvoDraughts figures 2–4 for the thesis report.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="standardized_test_results_evolution_match_*.csv (default: latest in results/)",
    )
    parser.add_argument(
        "--figure4-files",
        nargs="*",
        default=None,
        help="Optional: specific evolution_*.txt basenames under results/. "
        "Default: all non-minimax evolution_*.txt with generation stats.",
    )
    args = parser.parse_args()

    csv_path = args.csv or _find_latest_standardized_csv()
    if csv_path is None:
        raise SystemExit(f"No standardized_test_results_evolution_match_*.csv in {RESULTS_DIR}")

    rows = _load_csv_rows(csv_path)
    source_tag = csv_path.name
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    out2 = ANALYSIS_DIR / "figure_2_train_vs_test_scatter.png"
    out3 = ANALYSIS_DIR / "figure_3_8x8_test_win_rate_by_condition.png"
    out4 = ANALYSIS_DIR / "figure_4_fitness_over_generations.png"

    plot_figure2_train_vs_test(rows, source_tag, out2)
    print(out2)

    plot_figure3_conditions(rows, source_tag, out3)
    print(out3)

    if args.figure4_files:
        f4_paths = [RESULTS_DIR / name for name in args.figure4_files]
    else:
        f4_paths = discover_evolution_8x8_files()
    plot_figure4_fitness_curves(f4_paths, out4)
    print(out4)


if __name__ == "__main__":
    main()
