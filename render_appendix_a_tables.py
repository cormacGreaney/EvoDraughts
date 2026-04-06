"""
Render Appendix A experiment tables as PNG images for the report.
Regenerate: python docs/render_appendix_a_tables.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table


def _measure_max_text_width_inches(
    col_labels: list[str],
    rows: list[list[str]],
    fontsize: float,
    header_size: float,
    measure_dpi: float,
) -> list[float]:
    
    fig, ax = plt.subplots(figsize=(3, 2), dpi=measure_dpi)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    ncols = len(col_labels)
    col_inches: list[float] = []
    for j in range(ncols):
        max_px = 0.0
        t = ax.text(
            0.0,
            0.5,
            col_labels[j],
            fontsize=header_size,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.canvas.draw()
        max_px = max(max_px, float(t.get_window_extent(renderer=renderer).width))
        t.remove()
        for row in rows:
            t = ax.text(0.0, 0.5, row[j], fontsize=fontsize, ha="left", va="center")
            fig.canvas.draw()
            max_px = max(max_px, float(t.get_window_extent(renderer=renderer).width))
            t.remove()
        # display pixels → inches for this figure
        col_inches.append(max_px / fig.dpi)
    plt.close(fig)
    return col_inches


def _save_table(
    col_labels: list[str],
    rows: list[list[str]],
    out_path: Path,
    height_inches: float,
    fontsize: float,
    header_size: float | None = None,
    cell_pad: float = 0.002,
    col_pad_inches: float = 0.028,
    measure_dpi: float = 150,
    margin_w_inches: float = 0.12,
    pad_inches: float = 0.02,
) -> None:
    header_size = header_size or fontsize + 0.5
    widths_in = _measure_max_text_width_inches(
        col_labels, rows, fontsize, header_size, measure_dpi
    )
    padded = [w + col_pad_inches for w in widths_in]
    s = sum(padded)
    col_widths = [w / s for w in padded]

    fig_w = s + margin_w_inches

    fig, ax = plt.subplots(figsize=(fig_w, height_inches), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    table: Table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="upper left",
        cellLoc="left",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.scale(1.0, 1.22)

    for (row, col), cell in table.get_celld().items():
        cell.PAD = cell_pad
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=header_size, ha="left")
            cell.set_facecolor("#e8e8e8")
        else:
            cell.set_text_props(fontsize=fontsize, ha="left")
            if row % 2 == 0:
                cell.set_facecolor("#f5f5f5")

    fig.subplots_adjust(0, 0, 1, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=pad_inches,
        dpi=300,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent

    table_a1_cols = [
        "ID",
        "Result file",
        "Configuration",
        "Pop.",
        "Gens",
        "Games / eval",
        "Mutation",
        "Co-ev. ratio",
        "Train WR",
        "Test WR",
        "95% CI (test)",
        "Train–test gap",
        "Var (seeds)",
    ]
    table_a1_rows = [
        ["6C-1", "evolution_20251213_034131.txt", "6×6 classic", "100", "10", "10", "0.05", "Random only", "70.0%", "43.3%", "35.4%–51.3%", "26.7%", "0.112"],
        ["6C-2", "evolution_20251213_055102.txt", "6×6 classic", "200", "100", "20", "0.05", "Random only", "72.0%", "53.3%", "45.4%–61.3%", "19.2%", "0.070"],
        ["6C-3", "evolution_20251213_083718.txt", "6×6 classic", "200", "100", "20", "0.05", "Random only", "68.0%", "47.3%", "39.3%–55.3%", "20.2%", "0.093"],
        ["6I-1", "evolution_20251214_220441.txt", "6×6 improved", "150", "30", "15", "0.10", "60% / 40%", "80.0%", "32.0%", "24.5%–39.5%", "48.0%", "0.098"],
        ["6I-2", "evolution_20251215_032724.txt", "6×6 improved", "300", "50", "25", "0.10", "60% / 40%", "100.0%", "84.7%", "78.9%–90.4%", "15.3%", "0.075"],
        ["6I-3", "evolution_20251215_090842.txt", "6×6 improved", "300", "50", "25", "0.10", "60% / 40%", "100.0%", "34.0%", "26.4%–41.6%", "66.0%", "0.049"],
        ["6I-4", "evolution_20251215_202804.txt", "6×6 improved", "300", "50", "25", "0.10", "60% / 40%", "77.0%", "41.3%", "33.5%–49.2%", "35.7%", "0.091"],
        ["6I-5", "evolution_20251216_034713.txt", "6×6 improved", "300", "50", "25", "0.10", "60% / 40%", "82.0%", "81.3%", "75.1%–87.6%", "0.7%", "0.034"],
    ]

    table_a2_cols = [
        "ID",
        "Result file",
        "Condition",
        "Evo. seed",
        "Mut.",
        "Co-ev. / rnd",
        "Train WR",
        "Test WR",
        "95% CI (test)",
        "Train–test gap",
        "Var (seeds)",
    ]
    em = "\u2014"
    table_a2_rows = [
        ["8-01", "evolution_8x8_20251216_222007.txt", "Config_5 (pilot)", em, "0.10", "60% / 40%", "85.0%", "44.7%", "36.7%–52.6%", "40.3%", "0.050"],
        ["8-02", "evolution_8x8_20251218_002100.txt", "Baseline Config_6", "123", "0.15", "40% / 60%", "82.0%", "72.7%", "65.5%–79.8%", "9.8%", "0.068"],
        ["8-03", "evolution_8x8_20260216_052949.txt", "Baseline rep. 1", "456", "0.15", "40% / 60%", "88.0%", "78.0%", "71.4%–84.6%", "9.5%", "0.034"],
        ["8-04", "evolution_8x8_20260218_003143.txt", "Baseline rep. 2", "789", "0.15", "40% / 60%", "93.0%", "82.7%", "76.6%–88.7%", "9.8%", "0.025"],
        ["8-05", "evolution_8x8_20260219_203646.txt", "Baseline rep. 3", "1011", "0.15", "40% / 60%", "95.0%", "94.7%", "91.1%–98.3%", "0.3%", "0.034"],
        ["8-06", "evolution_8x8_20260223_045614.txt", "Cond. A (30% co-ev.)", "2001", "0.15", "30% / 70%", "93.0%", "76.0%", "69.2%–82.8%", "16.5%", "0.061"],
        ["8-07", "evolution_8x8_20260225_114105.txt", "Cond. A rep. 2", "2002", "0.15", "30% / 70%", "90.0%", "69.3%", "62.0%–76.7%", "20.7%", "0.013"],
        ["8-08", "evolution_8x8_20260227_011809.txt", "Cond. A rep. 3", "2003", "0.15", "30% / 70%", "90.0%", "66.7%", "59.1%–74.2%", "23.3%", "0.063"],
        ["8-09", "evolution_8x8_20260302_115155.txt", "Cond. B (50% co-ev.)", "2011", "0.15", "50% / 50%", "93.0%", "72.0%", "64.8%–79.2%", "20.5%", "0.027"],
        ["8-10", "evolution_8x8_20260307_093010.txt", "Cond. B rep. 2", "2012", "0.15", "50% / 50%", "95.0%", "73.3%", "66.3%–80.4%", "21.7%", "0.092"],
        ["8-11", "evolution_8x8_20260309_155326.txt", "Cond. B rep. 3", "2013", "0.15", "50% / 50%", "95.0%", "81.3%", "75.1%–87.6%", "13.7%", "0.062"],
        ["8-12", "evolution_8x8_20260311_043951.txt", "Cond. C (mut. 0.10)", "3011", "0.10", "40% / 60%", "95.0%", "78.0%", "71.4%–84.6%", "17.0%", "0.050"],
        ["8-13", "evolution_8x8_20260312_214315.txt", "Cond. C rep. 2", "3012", "0.10", "40% / 60%", "93.0%", "70.0%", "62.7%–77.3%", "22.5%", "0.087"],
        ["8-14", "evolution_8x8_20260316_010459.txt", "Cond. C rep. 3", "3013", "0.10", "40% / 60%", "97.0%", "63.3%", "55.6%–71.1%", "34.2%", "0.092"],
        ["8-15", "evolution_8x8_20260318_051106.txt", "Cond. D (mut. 0.20)", "3021", "0.20", "40% / 60%", "90.0%", "78.0%", "71.4%–84.6%", "12.0%", "0.034"],
        ["8-16", "evolution_8x8_20260320_104123.txt", "Cond. D rep. 2", "3022", "0.20", "40% / 60%", "93.0%", "81.3%", "75.1%–87.6%", "11.2%", "0.050"],
        ["8-17", "evolution_8x8_20260322_094426.txt", "Cond. D rep. 3", "3023", "0.20", "40% / 60%", "90.0%", "68.7%", "61.2%–76.1%", "21.3%", "0.072"],
    ]

    _save_table(
        table_a1_cols,
        table_a1_rows,
        root / "appendix_a_table_a1_6x6.png",
        height_inches=4.0,
        fontsize=7.0,
        header_size=7.25,
    )
    _save_table(
        table_a2_cols,
        table_a2_rows,
        root / "appendix_a_table_a2_8x8.png",
        height_inches=9.2,
        fontsize=7.0,
        header_size=7.25,
    )
    print(f"Wrote:\n  {root / 'appendix_a_table_a1_6x6.png'}\n  {root / 'appendix_a_table_a2_8x8.png'}")


if __name__ == "__main__":
    main()
