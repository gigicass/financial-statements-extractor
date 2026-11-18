"""Compare extracted values against the CVM ground truth.

Level 1 of the evaluation: standardized key totals (CVM fixed codes that every company
reports, like Ativo Total or Patrimônio Líquido). They are robust to differences between
the PDF layout and the CVM chart of accounts, so they work for any company or model.
"""

from decimal import Decimal

import pandas as pd

from src.extraction.pdf_reader import normalize
from src.extraction.schemas import FinancialStatement, StatementType

# CVM code -> accepted names of the matching total line in the PDF (normalized).
# Careful: CVM "2 - Passivo Total" is liabilities PLUS equity, i.e. the PDF line
# "Total do Passivo e Patrimônio Líquido", not "Total do Passivo".
KEY_TOTALS: dict[StatementType, dict[str, list[str]]] = {
    StatementType.BALANCE_SHEET: {
        "1": ["total do ativo", "ativo total"],
        "1.01": ["ativo circulante", "total do ativo circulante"],
        "1.02": ["ativo nao circulante", "total do ativo nao circulante"],
        "2": ["total do passivo e patrimonio liquido", "total do passivo e do patrimonio liquido",
              "passivo total"],
        "2.01": ["passivo circulante", "total do passivo circulante"],
        "2.02": ["passivo nao circulante", "total do passivo nao circulante"],
        "2.03": ["total do patrimonio liquido", "patrimonio liquido",
                 "patrimonio liquido consolidado"],
    },
}


def compare_values(extracted: Decimal, expected: Decimal) -> dict:
    """Exact match plus absolute and percentage error (both in BRL units)."""
    abs_error = abs(extracted - expected)
    pct_error = abs_error / abs(expected) * 100 if expected != 0 else None
    return {"exact_match": abs_error == 0, "abs_error": abs_error, "pct_error": pct_error}


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
        if line:
            row.update(compare_values(line.value * multiplier, expected))
        else:
            row.update({"exact_match": False, "abs_error": None, "pct_error": None})
        rows.append(row)
    return pd.DataFrame(rows)


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
