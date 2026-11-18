"""Map extracted PDF lines to CVM account codes.

The CVM DFP uses a standardized chart of accounts, while the PDF follows each company's
own layout, so names alone are not enough. The line-level evaluation therefore uses a
reference mapping file per company/statement: PDF account -> CVM code(s), reviewed by hand.

This module (1) drafts that file automatically and (2) loads a reviewed file and pairs
its rows with the lines of a new extraction. The draft uses values to suggest codes,
which is fine for a *draft to be reviewed*, but the evaluation itself must only use the
reviewed mapping - otherwise a wrong value could never be detected.

Usage (draft):
    uv run python -m src.evaluation.matching \
        data/processed/WEG_DFP_2025_balance_sheet_consolidated.json --company WEG
"""

import argparse
import csv
import difflib
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import COMPANIES, DEFAULT_CVM_ZIP, load_cvm_statement
from src.extraction.extractor import load_statement
from src.extraction.pdf_reader import normalize
from src.extraction.schemas import FinancialStatement, StatementLine, StatementType

logger = logging.getLogger(__name__)

MAPPINGS_DIR = Path("mappings")
MAPPING_COLUMNS = ["section", "pdf_account_name", "pdf_value", "is_total", "cvm_codes",
                   "cvm_names", "draft_status", "reviewed"]

# Names in a new extraction may differ slightly from the reviewed file (e.g. another
# model fixes a typo or changes case). Below this similarity a line is not paired.
MIN_NAME_SIMILARITY = 0.85

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


@dataclass
class MappingRow:
    """One reviewed row: a PDF account and the signed CVM codes that make up its value."""

    section: str
    pdf_account_name: str
    codes: list[tuple[int, str]]  # (+1 or -1, CVM code)
    is_total: bool = False


def mapping_path(company: str, statement: FinancialStatement) -> Path:
    """Reviewed mapping file name, e.g. mappings/WEG_2025_balance_sheet_consolidated.csv."""
    return MAPPINGS_DIR / (f"{company}_{statement.reference_date.year}_"
                           f"{statement.statement_type.value}_{statement.scope.value}.csv")


def parse_codes(expression: str) -> list[tuple[int, str]]:
    """'2.01+2.02' -> [(1,'2.01'),(1,'2.02')]; '2.03-2.03.09' -> [(1,'2.03'),(-1,'2.03.09')]."""
    tokens = re.findall(r"([+-]?)\s*(\d+(?:\.\d+)*)", expression)
    return [(-1 if sign == "-" else 1, code) for sign, code in tokens]


# ---------------------------------------------------------------- sections


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


def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


# ---------------------------------------------------------------- draft


def _is_ancestor(a: str, b: str) -> bool:
    return b.startswith(a + ".")


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
            "is_total": str(line.is_total),
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
    logger.info("Mapping written to %s", path)


# ---------------------------------------------------------------- reviewed mapping


def load_mapping(path: Path) -> list[MappingRow]:
    """Load a reviewed mapping file. Rows with empty cvm_codes are kept (not evaluated)."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [MappingRow(section=r["section"], pdf_account_name=r["pdf_account_name"],
                           codes=parse_codes(r["cvm_codes"] or ""),
                           is_total=r.get("is_total") == "True")
                for r in csv.DictReader(f)]


def pair_lines(statement: FinancialStatement, mapping: list[MappingRow],
               ) -> tuple[list[tuple[MappingRow, StatementLine | None]], list[StatementLine]]:
    """Pair each mapping row with an extracted line of the same section.

    1st pass: same or similar name.
    2nd pass, subtotals only: by position among the section's subtotals. Some subtotals
    have no label in the PDF, so each model names them differently ('Realizável a longo
    prazo' vs 'Não circulante'); their position is the stable identity. Values are never
    used for pairing, otherwise a wrong value could not be detected.

    Returns (pairs, extra_lines): extra lines are extracted lines no mapping row claimed,
    e.g. invented lines or accounts the reference file does not have.
    """
    available = list(zip(statement.lines, assign_sections(statement)))
    used: set[int] = set()
    paired: dict[int, StatementLine] = {}  # mapping row index -> extracted line
    for row_index, row in enumerate(mapping):
        candidates = [(i, line) for i, (line, section) in enumerate(available)
                      if i not in used and (section or "") == row.section]
        best = max(candidates, default=None,
                   key=lambda c: _name_similarity(c[1].account_name, row.pdf_account_name))
        if best and _name_similarity(best[1].account_name,
                                     row.pdf_account_name) >= MIN_NAME_SIMILARITY:
            used.add(best[0])
            paired[row_index] = best[1]

    for section in {row.section for row in mapping}:
        row_totals = [k for k, row in enumerate(mapping)
                      if row.is_total and row.section == section]
        line_totals = [i for i, (line, s) in enumerate(available)
                       if line.is_total and (s or "") == section]
        for position, row_index in enumerate(row_totals):
            if row_index in paired or position >= len(line_totals):
                continue
            if line_totals[position] not in used:
                used.add(line_totals[position])
                paired[row_index] = available[line_totals[position]][0]

    pairs = [(row, paired.get(k)) for k, row in enumerate(mapping)]
    extras = [line for i, (line, _) in enumerate(available) if i not in used]
    return pairs, extras


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft a PDF -> CVM mapping for review.")
    parser.add_argument("extracted_json", type=Path)
    parser.add_argument("--company", required=True, choices=sorted(COMPANIES))
    parser.add_argument("--cvm-zip", type=Path, default=DEFAULT_CVM_ZIP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    statement = load_statement(args.extracted_json)
    cvm = load_cvm_statement(args.cvm_zip, COMPANIES[args.company],
                             statement.statement_type, statement.scope)
    out = mapping_path(args.company, statement).with_suffix(".draft.csv")
    if out.exists():
        raise FileExistsError(f"{out} already exists; delete it to regenerate the draft")
    write_mapping(draft_mapping(statement, cvm), out)


if __name__ == "__main__":
    main()
