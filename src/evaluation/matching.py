"""Map extracted PDF lines to CVM account codes.

The CVM DFP uses a standardized chart of accounts, while the PDF follows each company's
own layout, so names alone are not enough (see README). The evaluation therefore uses a
reference mapping file per company/statement: PDF account -> CVM code(s), reviewed by hand.

This module drafts that file automatically. The draft uses values to suggest codes,
which is fine for a *draft to be reviewed*, but the evaluation itself must only use the
reviewed mapping - otherwise a wrong value could never be detected.

Usage:
    uv run python -m src.evaluation.matching \
        data/processed/WEG_DFP_2025_balance_sheet_consolidated.json --company WEG
"""

import argparse
import csv
import difflib
import logging
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import COMPANIES, load_cvm_statement
from src.evaluation.evaluate import DEFAULT_CVM_ZIP, load_extracted
from src.evaluation.metrics import KEY_TOTALS
from src.extraction.pdf_reader import normalize
from src.extraction.schemas import FinancialStatement

logger = logging.getLogger(__name__)

MAPPINGS_DIR = Path("mappings")
MAPPING_COLUMNS = ["section", "pdf_account_name", "pdf_value", "cvm_codes",
                   "cvm_names", "draft_status", "reviewed"]


def assign_sections(statement: FinancialStatement) -> list[str | None]:
    """CVM section code for each line, inferred from the totals that follow it.

    Lines are buffered until a key total (e.g. 'Ativo Circulante' -> '1.01') closes them.
    """
    alias_to_code = {alias: code
                     for code, aliases in KEY_TOTALS[statement.statement_type].items()
                     for alias in aliases}
    sections: list[str | None] = [None] * len(statement.lines)
    pending: list[int] = []
    for i, line in enumerate(statement.lines):
        code = alias_to_code.get(normalize(line.account_name)) if line.is_total else None
        if code is None:
            pending.append(i)
            continue
        sections[i] = code
        for j in pending:
            sections[j] = code
        pending = []
    return sections


def _is_ancestor(a: str, b: str) -> bool:
    return b.startswith(a + ".")


def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def suggest_codes(name: str, value: Decimal, section: str | None,
                  cvm: pd.DataFrame) -> tuple[list[str], str]:
    """Suggest CVM code(s) for one PDF line. Returns (codes, draft_status)."""
    if value == 0:
        return [], "zero_in_pdf"
    for scope in ([section] if section else []) + [None]:
        pool = cvm if scope is None else cvm[(cvm["code"] == scope)
                                             | cvm["code"].str.startswith(scope + ".")]
        pool = pool[pool["value"] != 0]
        # 1) single account with the same value; ties broken by name similarity
        same = pool[pool["value"] == value]
        if not same.empty:
            best = max(same.itertuples(), key=lambda r: _name_similarity(name, r.name))
            return [best.code], "single" if len(same) == 1 else "single_tiebreak_by_name"
        # 2) sum of two accounts (neither one inside the other)
        for a, b in combinations(pool.itertuples(), 2):
            if a.value + b.value == value and not (_is_ancestor(a.code, b.code)
                                                   or _is_ancestor(b.code, a.code)):
                return [a.code, b.code], "sum_of_two"
    return [], "not_found"


def draft_mapping(statement: FinancialStatement, cvm: pd.DataFrame) -> list[dict]:
    """One draft mapping row per extracted line."""
    multiplier = statement.value_multiplier()
    names = dict(zip(cvm["code"], cvm["name"]))
    rows = []
    for line, section in zip(statement.lines, assign_sections(statement)):
        codes, status = suggest_codes(line.account_name, line.value * multiplier, section, cvm)
        rows.append({
            "section": section or "",
            "pdf_account_name": line.account_name,
            "pdf_value": str(line.value),
            "cvm_codes": "+".join(codes),
            "cvm_names": " + ".join(names[c] for c in codes),
            "draft_status": status,
            "reviewed": "",
        })
    return rows


def write_mapping(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM so Excel shows accents correctly.
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Draft mapping written to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft a PDF -> CVM mapping for review.")
    parser.add_argument("extracted_json", type=Path)
    parser.add_argument("--company", required=True, choices=sorted(COMPANIES))
    parser.add_argument("--cvm-zip", type=Path, default=DEFAULT_CVM_ZIP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    statement = load_extracted(args.extracted_json)
    cvm = load_cvm_statement(args.cvm_zip, COMPANIES[args.company],
                             statement.statement_type, statement.scope)
    out = MAPPINGS_DIR / (f"{args.company}_{statement.reference_date.year}_"
                          f"{statement.statement_type.value}_{statement.scope.value}.draft.csv")
    if out.exists():
        raise FileExistsError(f"{out} already exists; delete it to regenerate the draft")
    write_mapping(draft_mapping(statement, cvm), out)


if __name__ == "__main__":
    main()
