"""
Plot standardized test win rates from the latest
standardized_test_results_evolution_match_*.csv (output of
reevaluate_strategies_match_evolution.py)

Writes one PNG under results/analysis/
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"


def _find_latest_evolution_match_csv() -> Path | None:
    files = list(RESULTS_DIR.glob("standardized_test_results_evolution_match_*.csv"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _short_label(result_file: str) -> str:
    name = result_file.replace(".txt", "")
    if name.startswith("evolution_"):
        name = name[len("evolution_") :]
    if len(name) > 40:
        return name[:37] + "..."
    return name


def main() -> None:
    csv_path = _find_latest_evolution_match_csv()
    if csv_path is None:
        raise SystemExit(
            f"No standardized_test_results_evolution_match_*.csv in {RESULTS_DIR}"
        )

    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("Result File", "")
            if fn and "minimax" in fn.lower():
                continue
            rows.append(row)

    if not rows:
        raise SystemExit(f"CSV is empty or only minimax rows: {csv_path}")

    # Sort: 6x6 first, then 8x8; within each by test win rate descending
    def sort_key(r: dict[str, str]) -> tuple[int, float]:
        bs = r.get("Board Size", "")
        board_order = 0 if bs == "6x6" else 1
        try:
            wr = float(r["Test Win Rate"])
        except (KeyError, ValueError):
            wr = 0.0
        return (board_order, -wr)

    rows.sort(key=sort_key)

    labels = [_short_label(r["Result File"]) for r in rows]
    rates = np.array([float(r["Test Win Rate"]) * 100.0 for r in rows])

    colors = [
        "#3498db" if r.get("Board Size") == "6x6" else "#27ae60"
        for r in rows
    ]

    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig_h = max(6.0, 0.28 * n + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_pos = np.arange(n)
    ax.barh(y_pos, rates, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Test win rate vs random (%)")
    ax.set_xlim(0, 105)
    ax.axvline(50.0, color="gray", linestyle="--", linewidth=1, alpha=0.8, label="50% (random)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    stem = csv_path.stem  # standardized_test_results_evolution_match_YYYYMMDD_HHMMSS
    ax.set_title(
        f"Standardized Test Performance\nSource: {csv_path.name}",
        fontsize=11,
    )
    fig.tight_layout()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{stem}_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
