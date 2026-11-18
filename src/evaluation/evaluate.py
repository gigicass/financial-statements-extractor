"""Evaluate an extracted statement (JSON from the extractor) against CVM data.

Usage:
    uv run python -m src.evaluation.evaluate \
        data/processed/WEG_DFP_2025_balance_sheet_consolidated.json --company WEG
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import COMPANIES, DEFAULT_CVM_ZIP, load_cvm_statement
from src.evaluation.matching import load_mapping, mapping_path
from src.evaluation.metrics import evaluate_key_totals, evaluate_lines, summarize
from src.extraction.extractor import load_statement

logger = logging.getLogger(__name__)


def _print_table(df: pd.DataFrame, columns: list[str]) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.max_colwidth", 50):
        print(df[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an extraction against CVM data.")
    parser.add_argument("extracted_json", type=Path)
    parser.add_argument("--company", required=True, choices=sorted(COMPANIES))
    parser.add_argument("--cvm-zip", type=Path, default=DEFAULT_CVM_ZIP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    statement = load_statement(args.extracted_json)
    cvm = load_cvm_statement(args.cvm_zip, COMPANIES[args.company],
                             statement.statement_type, statement.scope)

    print("\n== Key totals ==")
    totals = evaluate_key_totals(statement, cvm)
    _print_table(totals, ["code", "cvm_name", "extracted_name", "cvm_value",
                          "extracted_value", "exact_match"])
    print(summarize(totals))

    path = mapping_path(args.company, statement)
    if not path.exists():
        logger.warning("No reviewed mapping at %s; skipping line-level evaluation", path)
        return

    print("\n== Line level (reviewed mapping) ==")
    lines, extras = evaluate_lines(statement, load_mapping(path), cvm)
    errors = lines[~lines["exact_match"]]
    if errors.empty:
        print("All mapped accounts found with the exact CVM value.")
    else:
        _print_table(errors, ["section", "account", "cvm_codes", "cvm_value",
                              "extracted_value", "abs_error", "pct_error"])
    print(summarize(lines))
    print("Extracted lines not in the mapping:", extras or "none")


if __name__ == "__main__":
    main()
