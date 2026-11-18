"""Compare extracted values against the CVM ground truth.

Two levels:
- Key totals: standardized CVM codes every company reports (Ativo Total, PL...).
  Robust to layout differences, so they work for any company or model.
- Line level: every PDF account, through the hand-reviewed mapping (see matching.py).
"""

from decimal import Decimal

import pandas as pd

from src.evaluation.matching import KEY_TOTALS, MappingRow, find_key_total_line, pair_lines
from src.extraction.schemas import FinancialStatement, StatementLine


def compare_values(extracted: Decimal, expected: Decimal,
                   rounding_tolerance: Decimal = Decimal(0)) -> dict:
    """Exact match, match within rounding, and absolute/percentage error (BRL units).

    The CVM form rounds each account separately, so a PDF line that maps to a sum of
    CVM accounts can differ by 1 unit of the statement scale (e.g. R$ 1 mil) even when
    the value was read correctly. `within_rounding` tells those cases apart from real
    reading errors; `exact_match` stays the headline metric.
    """
    abs_error = abs(extracted - expected)
    pct_error = ((abs_error / abs(expected) * 100).quantize(Decimal("0.0001"))
                 if expected != 0 else None)
    return {"exact_match": abs_error == 0, "within_rounding": abs_error <= rounding_tolerance,
            "abs_error": abs_error, "pct_error": pct_error}


def _no_match() -> dict:
    return {"exact_match": False, "within_rounding": False, "abs_error": None,
            "pct_error": None}


def evaluate_key_totals(statement: FinancialStatement, cvm: pd.DataFrame) -> pd.DataFrame:
    """One row per key CVM total: was it extracted, and is the value exactly right?"""
    multiplier = statement.value_multiplier()
    cvm_by_code = cvm.set_index("code")

    rows = []
    for code in KEY_TOTALS[statement.statement_type]:
        expected = cvm_by_code.loc[code, "value"]
        line = find_key_total_line(statement, code)
        row = {"code": code, "cvm_name": cvm_by_code.loc[code, "name"], "cvm_value": expected,
               "extracted_name": line.account_name if line else None,
               "extracted_value": line.value * multiplier if line else None}
        row.update(compare_values(line.value * multiplier, expected, Decimal(multiplier))
                   if line else _no_match())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_lines(statement: FinancialStatement, mapping: list[MappingRow],
                   cvm: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Line-level evaluation through the reviewed mapping.

    Returns one row per mapped account (rows without CVM codes are skipped) and the
    names of extracted lines that no mapping row claimed ("extra" lines).
    """
    multiplier = statement.value_multiplier()
    cvm_values = dict(zip(cvm["code"], cvm["value"]))
    pairs, extras = pair_lines(statement, mapping)

    # Rows of the same group are evaluated together, as one account.
    units: dict[str, list[tuple[MappingRow, StatementLine | None]]] = {}
    for k, (mapping_row, line) in enumerate(pairs):
        if mapping_row.codes:
            units.setdefault(mapping_row.group or f"row-{k}", []).append((mapping_row, line))

    rows = []
    for members in units.values():
        first = members[0][0]
        expected = sum((sign * cvm_values[code] for sign, code in first.codes), Decimal(0))
        found = all(line is not None for _, line in members)
        extracted = (sum((line.value for _, line in members), Decimal(0)) * multiplier
                     if found else None)
        row = {"section": first.section,
               "account": " + ".join(r.pdf_account_name for r, _ in members),
               "cvm_codes": "".join(("-" if s < 0 else "+") + c
                                    for s, c in first.codes).lstrip("+"),
               "cvm_value": expected, "extracted_value": extracted}
        row.update(compare_values(extracted, expected, Decimal(multiplier))
                   if found else _no_match())
        rows.append(row)
    return pd.DataFrame(rows), [line.account_name for line in extras]


def summarize(results: pd.DataFrame) -> dict:
    """Aggregate metrics: coverage (found) and accuracy among the accounts found."""
    found = results["extracted_value"].notna()

    def pct_of_found(column: str) -> float:
        return round(100 * float(results.loc[found, column].mean()), 1) if found.any() else 0.0

    return {
        "accounts": len(results),
        "found": int(found.sum()),
        "coverage_pct": round(100 * float(found.mean()), 1),
        "exact_match_pct": pct_of_found("exact_match"),
        "within_rounding_pct": pct_of_found("within_rounding"),
    }
