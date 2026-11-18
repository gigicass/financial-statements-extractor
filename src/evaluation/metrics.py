"""Compare extracted values against the CVM ground truth.

Two levels:
- Key totals: standardized CVM codes every company reports (Ativo Total, PL...).
  Robust to layout differences, so they work for any company or model.
- Line level: every PDF account, through the hand-reviewed mapping (see matching.py).
"""

from decimal import Decimal

import pandas as pd

from src.evaluation.matching import KEY_TOTALS, MappingRow, pair_lines
from src.extraction.pdf_reader import normalize
from src.extraction.schemas import FinancialStatement


def compare_values(extracted: Decimal, expected: Decimal) -> dict:
    """Exact match plus absolute and percentage error (both in BRL units)."""
    abs_error = abs(extracted - expected)
    pct_error = ((abs_error / abs(expected) * 100).quantize(Decimal("0.0001"))
                 if expected != 0 else None)
    return {"exact_match": abs_error == 0, "abs_error": abs_error, "pct_error": pct_error}


def _no_match() -> dict:
    return {"exact_match": False, "abs_error": None, "pct_error": None}


def evaluate_key_totals(statement: FinancialStatement, cvm: pd.DataFrame) -> pd.DataFrame:
    """One row per key CVM total: was it extracted, and is the value exactly right?"""
    multiplier = statement.value_multiplier()
    totals = {normalize(line.account_name): line for line in statement.lines if line.is_total}
    cvm_by_code = cvm.set_index("code")

    rows = []
    for code, aliases in KEY_TOTALS[statement.statement_type].items():
        expected = cvm_by_code.loc[code, "value"]
        line = next((totals[a] for a in aliases if a in totals), None)
        row = {"code": code, "cvm_name": cvm_by_code.loc[code, "name"], "cvm_value": expected,
               "extracted_name": line.account_name if line else None,
               "extracted_value": line.value * multiplier if line else None}
        row.update(compare_values(line.value * multiplier, expected) if line else _no_match())
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

    rows = []
    for mapping_row, line in pairs:
        if not mapping_row.codes:
            continue
        expected = sum((sign * cvm_values[code] for sign, code in mapping_row.codes),
                       Decimal(0))
        extracted = line.value * multiplier if line else None
        row = {"section": mapping_row.section, "account": mapping_row.pdf_account_name,
               "cvm_codes": "".join(("-" if s < 0 else "+") + c
                                    for s, c in mapping_row.codes).lstrip("+"),
               "cvm_value": expected, "extracted_value": extracted}
        row.update(compare_values(extracted, expected) if line else _no_match())
        rows.append(row)
    return pd.DataFrame(rows), [line.account_name for line in extras]


def summarize(results: pd.DataFrame) -> dict:
    """Aggregate metrics: coverage (found) and accuracy (exact value among found)."""
    found = results["extracted_value"].notna()
    return {
        "accounts": len(results),
        "found": int(found.sum()),
        "coverage_pct": round(100 * float(found.mean()), 1),
        "exact_match_pct": round(100 * float(results.loc[found, "exact_match"].mean()), 1)
        if found.any() else 0.0,
    }
