"""Build the results tables (Markdown) and figures (PNG) from results/*.csv.

Every number in the README comes from this script, so the report is reproducible:
    uv run --group report python -m src.experiments.report

Outputs:
    docs/results.md            all tables
    docs/figures/*.png         figures used in the README
    results/error_injection.csv (computed once from local PDFs + CVM data, then reused)
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.experiments.tables import (  # noqa: E402
    balance_sheet_by_model,
    error_detection,
    hard_cases,
    income_statement_by_model,
    prompt_versions,
    strategies_with_feedback,
)

logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
FIGURES_DIR = DOCS_DIR / "figures"

# Reference palette (validated categorical slots 1-2) and text/surface tokens.
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- figures


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_cost(strategies: pd.DataFrame, path: Path) -> None:
    """Horizontal bars: cost per balance sheet for each strategy (single series)."""
    data = strategies.sort_values("_cost")
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE,
                           layout="constrained")
    _style(ax)
    bars = ax.barh(data["Strategy"], data["_cost"], height=0.5, color=SERIES_1)
    for bar, value in zip(bars, data["_cost"]):
        ax.text(bar.get_width() + 0.0008, bar.get_y() + bar.get_height() / 2,
                f"US$ {value:.3f}", va="center", color=TEXT, fontsize=10)
    ax.set_xlim(0, data["_cost"].max() * 1.25)
    ax.set_xlabel("Average cost per balance sheet (US$), retries included", color=TEXT_2)
    ax.tick_params(axis="y", colors=TEXT, labelsize=10)
    fig.suptitle("Every strategy ends with 12/12 balance sheets passing all checks;\n"
                 "the difference is cost", x=0.02, ha="left", color=TEXT, fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_detection(detection: pd.DataFrame, path: Path) -> None:
    """Grouped horizontal bars: % of planted errors detected, two detectors."""
    data = detection.iloc[::-1]
    y = range(len(data))
    h = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.0), facecolor=SURFACE,
                           layout="constrained")
    _style(ax)
    ax.barh([i + h / 2 + 0.02 for i in y], data["_integrity"], height=h, color=SERIES_1,
            label="Accounting checks (no ground truth)")
    ax.barh([i - h / 2 - 0.02 for i in y], data["_cvm"], height=h, color=SERIES_2,
            label="Comparison with CVM data")
    for i, (a, b) in enumerate(zip(data["_integrity"], data["_cvm"])):
        ax.text(a + 1.5, i + h / 2 + 0.02, f"{a:.0f}%", va="center", color=TEXT, fontsize=9)
        ax.text(b + 1.5, i - h / 2 - 0.02, f"{b:.0f}%", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(list(y), data["Planted error"], color=TEXT, fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Planted errors detected (%)", color=TEXT_2)
    fig.suptitle("Accounting checks catch every error that changes a sum,\n"
                 "but not two values swapped inside the same subtotal",
                 x=0.02, ha="left", color=TEXT, fontsize=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=9, labelcolor=TEXT, borderaxespad=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main


def _md(df: pd.DataFrame) -> str:
    return df[[c for c in df.columns if not c.startswith("_")]].to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build results tables and figures.")
    parser.add_argument("--refresh-injection", action="store_true",
                        help="recompute the error injection experiment (needs local data)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    strategies = strategies_with_feedback()
    detection = error_detection(args.refresh_injection)
    figure_cost(strategies, FIGURES_DIR / "cost_per_balance_sheet.png")
    figure_detection(detection, FIGURES_DIR / "error_detection.png")

    sections = [
        ("Balance sheet by model (4 companies x 3 repeats, prompt v2)", balance_sheet_by_model()),
        ("Retry with feedback when an accounting check fails", strategies),
        ("Prompt v1 vs v2 (first repeat)", prompt_versions()),
        ("Income statement key lines (6 companies, 1 run)", income_statement_by_model()),
        ("Hard cases (balance sheet)", hard_cases()),
        ("Planted errors: who detects them", detection),
    ]
    body = "\n\n".join(f"## {title}\n\n{_md(df)}" for title, df in sections)
    (DOCS_DIR / "results.md").write_text(
        "# Results\n\nGenerated by `src/experiments/report.py` from `results/*.csv`.\n\n"
        + body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
