"""Evaluate extracted statements (JSON from the extractor) against CVM data.

The company is inferred from the file name prefix (e.g. KLBN_DFP_2025_...json -> KLBN).

Usage:
    uv run python -m src.evaluation.evaluate data/processed/*_balance_sheet_consolidated.json
    uv run python -m src.evaluation.evaluate <file.json> --details   # list every error
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import COMPANIES, DEFAULT_CVM_ZIP, load_cvm_statement
from src.evaluation.integrity import check_file, summarize_checks
from src.evaluation.matching import load_mapping, mapping_path
from src.evaluation.metrics import evaluate_key_totals, evaluate_lines, summarize
from src.extraction.extractor import load_statement

logger = logging.getLogger(__name__)


def company_from_path(path: Path) -> str:
    """'KLBN_DFP_2025_balance_sheet_consolidated.json' -> 'KLBN'."""
    company = path.stem.split("_")[0]
    if company not in COMPANIES:
        raise ValueError(f"Unknown company '{company}' in {path.name}; known: {sorted(COMPANIES)}")
    return company


def _print_table(df: pd.DataFrame, columns: list[str]) -> None:
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_colwidth", 50):
        print(df[columns].to_string(index=False))


def evaluate_file(json_path: Path, cvm_zip: Path, details: bool) -> dict:
    """Run both evaluation levels for one extracted statement; return one summary row."""
    company = company_from_path(json_path)
    statement = load_statement(json_path)
    cvm = load_cvm_statement(cvm_zip, COMPANIES[company], statement.statement_type,
                             statement.scope)

    totals = evaluate_key_totals(statement, cvm)
    row = {"company": company, "statement": statement.statement_type.value}
    row.update({f"totals_{k}": v for k, v in summarize(totals).items()})
    row.update(summarize_checks(check_file(json_path)))

    path = mapping_path(company, statement)
    if not path.exists():
        logger.warning("No reviewed mapping at %s; skipping line-level evaluation", path)
        return row

    lines, extras = evaluate_lines(statement, load_mapping(path), cvm)
    row.update({f"lines_{k}": v for k, v in summarize(lines).items()})
    row["extra_lines"] = len(extras)

    if details:
        print(f"\n== {company}: key totals ==")
        _print_table(totals, ["code", "cvm_name", "extracted_name", "cvm_value",
                              "extracted_value", "exact_match"])
        print(f"\n== {company}: line-level differences ==")
        errors = lines[~lines["exact_match"]]
        if errors.empty:
            print("none")
        else:
            _print_table(errors, ["section", "account", "cvm_codes", "cvm_value",
                                  "extracted_value", "abs_error", "within_rounding"])
        print("Extracted lines not in the mapping:", extras or "none")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extractions against CVM data.")
    parser.add_argument("extracted_json", type=Path, nargs="+")
    parser.add_argument("--cvm-zip", type=Path, default=DEFAULT_CVM_ZIP)
    parser.add_argument("--details", action="store_true", help="print every difference")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    summary = pd.DataFrame([evaluate_file(p, args.cvm_zip, args.details)
                            for p in args.extracted_json])
    columns = ["company", "totals_found", "totals_accounts", "totals_exact_match_pct",
               "lines_accounts", "lines_coverage_pct", "lines_exact_match_pct",
               "lines_within_rounding_pct", "extra_lines", "checks_passed", "checks_failed",
               "checks_inconclusive"]
    print("\n== Summary ==")
    _print_table(summary, [c for c in columns if c in summary.columns])


if __name__ == "__main__":
    main()
