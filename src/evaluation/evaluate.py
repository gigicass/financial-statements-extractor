"""Evaluate an extracted statement (JSON from the extractor) against CVM data.

Usage:
    uv run python -m src.evaluation.evaluate \
        data/processed/WEG_DFP_2025_balance_sheet_consolidated.json --company WEG
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import COMPANIES, load_cvm_statement
from src.evaluation.metrics import evaluate_key_totals, summarize
from src.extraction.schemas import FinancialStatement

DEFAULT_CVM_ZIP = Path("data/raw/cvm/dfp_cia_aberta_2025.zip")


def load_extracted(json_path: Path) -> FinancialStatement:
    """Read a statement saved by the extractor."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return FinancialStatement.model_validate(data["statement"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an extraction against CVM data.")
    parser.add_argument("extracted_json", type=Path)
    parser.add_argument("--company", required=True, choices=sorted(COMPANIES))
    parser.add_argument("--cvm-zip", type=Path, default=DEFAULT_CVM_ZIP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    statement = load_extracted(args.extracted_json)
    cvm = load_cvm_statement(args.cvm_zip, COMPANIES[args.company],
                             statement.statement_type, statement.scope)
    results = evaluate_key_totals(statement, cvm)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results[["code", "cvm_name", "extracted_name", "cvm_value",
                       "extracted_value", "exact_match"]].to_string(index=False))
    print(summarize(results))


if __name__ == "__main__":
    main()
