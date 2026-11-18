"""Experiment: plant typical extraction errors and measure who detects them.

For each extracted statement, we inject one error at a time and ask two detectors:
- integrity checks (no ground truth needed), and
- the CVM comparison (ground truth, through the reviewed mapping).

No LLM calls: the errors are synthetic, so the experiment is free and reproducible.

Usage:
    uv run python -m src.evaluation.error_injection data/processed/*_balance_sheet_consolidated.json
"""

import argparse
import logging
import random
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import COMPANIES, DEFAULT_CVM_ZIP, load_cvm_statement
from src.evaluation.evaluate import company_from_path
from src.evaluation.integrity import run_checks, statement_pages_text
from src.evaluation.matching import load_mapping, mapping_path
from src.evaluation.metrics import evaluate_key_totals, evaluate_lines
from src.extraction.extractor import load_statement, source_pdf_for
from src.extraction.schemas import CurrencyScale, FinancialStatement

SEED = 42
TRIALS_PER_ERROR = 5

Injector = Callable[[FinancialStatement, random.Random], FinancialStatement | None]


def _detail_indexes(statement: FinancialStatement) -> list[int]:
    """Non-total lines with a non-zero value: the ones an error can affect."""
    return [i for i, line in enumerate(statement.lines) if not line.is_total and line.value != 0]


def _with_value(statement: FinancialStatement, index: int, value: Decimal) -> FinancialStatement:
    copy = statement.model_copy(deep=True)
    copy.lines[index].value = value
    return copy


def drop_line(statement, rng):
    """The model skips one account."""
    index = rng.choice(_detail_indexes(statement))
    copy = statement.model_copy(deep=True)
    del copy.lines[index]
    return copy


def swap_digits(statement, rng):
    """Two adjacent digits swapped, e.g. 9.911.053 -> 9.911.035."""
    index = rng.choice(_detail_indexes(statement))
    digits = list(str(abs(int(statement.lines[index].value))))
    positions = [p for p in range(len(digits) - 1) if digits[p] != digits[p + 1]]
    if not positions:
        return None
    p = rng.choice(positions)
    digits[p], digits[p + 1] = digits[p + 1], digits[p]
    sign = -1 if statement.lines[index].value < 0 else 1
    return _with_value(statement, index, sign * Decimal("".join(digits)))


def flip_sign(statement, rng):
    """A value in parentheses read as positive (or the other way round)."""
    index = rng.choice(_detail_indexes(statement))
    return _with_value(statement, index, -statement.lines[index].value)


def swap_neighbor_values(statement, rng):
    """Values of two consecutive accounts swapped: the subtotal still adds up."""
    details = set(_detail_indexes(statement))
    pairs = [i for i in details if i + 1 in details
             and statement.lines[i].value != statement.lines[i + 1].value]
    if not pairs:
        return None
    i = rng.choice(pairs)
    copy = statement.model_copy(deep=True)
    copy.lines[i].value, copy.lines[i + 1].value = (statement.lines[i + 1].value,
                                                    statement.lines[i].value)
    return copy


def wrong_scale(statement, rng):
    """Values right, but the scale declared as UNIDADE instead of MIL (or vice versa)."""
    copy = statement.model_copy(deep=True)
    copy.currency_scale = (CurrencyScale.UNIT if statement.currency_scale == CurrencyScale.THOUSAND
                           else CurrencyScale.THOUSAND)
    return copy


INJECTORS: dict[str, Injector] = {
    "drop_line": drop_line,
    "swap_digits": swap_digits,
    "flip_sign": flip_sign,
    "swap_neighbor_values": swap_neighbor_values,
    "wrong_scale": wrong_scale,
}


def _cvm_errors(statement, mapping, cvm) -> int:
    """Number of problems the CVM comparison reports (rounding differences excluded)."""
    lines, _ = evaluate_lines(statement, mapping, cvm)
    totals = evaluate_key_totals(statement, cvm)
    found_totals = totals[totals["extracted_value"].notna()]
    return int((~lines["within_rounding"]).sum() + (~found_totals["within_rounding"]).sum())


def run_experiment(json_paths: list[Path], cvm_zip: Path) -> pd.DataFrame:
    rows = []
    for json_path in json_paths:
        company = company_from_path(json_path)
        original = load_statement(json_path)
        cvm = load_cvm_statement(cvm_zip, COMPANIES[company], original.statement_type,
                                 original.scope)
        mapping = load_mapping(mapping_path(company, original))
        pages_text = statement_pages_text(original, source_pdf_for(json_path))

        # Baselines: the experiment counts only what the injected error changes.
        base_checks = sum(r.passed is False for r in run_checks(original, pages_text))
        base_cvm = _cvm_errors(original, mapping, cvm)

        for error_name, inject in INJECTORS.items():
            rng = random.Random(f"{SEED}-{company}-{error_name}")
            trials = 1 if error_name == "wrong_scale" else TRIALS_PER_ERROR
            for _ in range(trials):
                corrupted = inject(original, rng)
                if corrupted is None:
                    continue
                failed = sum(r.passed is False for r in run_checks(corrupted, pages_text))
                rows.append({
                    "company": company, "error": error_name,
                    "integrity_detected": failed > base_checks,
                    "cvm_detected": _cvm_errors(corrupted, mapping, cvm) > base_cvm,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plant errors and measure detection.")
    parser.add_argument("extracted_json", type=Path, nargs="+")
    parser.add_argument("--cvm-zip", type=Path, default=DEFAULT_CVM_ZIP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    results = run_experiment(args.extracted_json, args.cvm_zip)
    table = results.groupby("error").agg(
        trials=("error", "size"),
        integrity_detected=("integrity_detected", "sum"),
        cvm_detected=("cvm_detected", "sum"),
    )
    table.loc["TOTAL"] = table.sum()
    table["integrity_pct"] = (100 * table["integrity_detected"] / table["trials"]).round(1)
    table["cvm_pct"] = (100 * table["cvm_detected"] / table["trials"]).round(1)
    print(table.to_string())


if __name__ == "__main__":
    main()
