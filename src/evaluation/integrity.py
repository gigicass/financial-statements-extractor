"""Accounting checks that need NO ground truth: only the statement's internal coherence.

These are what an extractor in production would use to verify itself, for any company
(even ones not in the CVM dataset). They cannot catch everything - e.g. two values swapped
inside the same subtotal still add up - which is why the CVM comparison remains.

Usage:
    uv run python -m src.evaluation.integrity data/processed/*_balance_sheet_consolidated.json
"""

import argparse
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from src.evaluation.matching import find_key_total_line
from src.extraction.extractor import load_statement, source_pdf_for
from src.extraction.pdf_reader import find_statement_pages, normalize, read_pages
from src.extraction.schemas import CurrencyScale, FinancialStatement, StatementType

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Outcome of one check. passed=None means inconclusive (could not be evaluated)."""

    name: str
    passed: bool | None
    detail: str


def _fmt(value: Decimal) -> str:
    """Brazilian thousands separator, as printed in the statements: 26.910.845."""
    return f"{value:,}".replace(",", ".")


def _find_key_total(statement: FinancialStatement, code: str) -> Decimal | None:
    """Value of the total line that corresponds to a CVM key code (e.g. '1' = Ativo Total)."""
    line = find_key_total_line(statement, code)
    return line.value if line else None


def check_balance_equation(statement: FinancialStatement) -> CheckResult:
    """Ativo Total = Passivo + Patrimônio Líquido."""
    name = "Ativo = Passivo + PL"
    assets, liabilities_equity = _find_key_total(statement, "1"), _find_key_total(statement, "2")
    if assets is None or liabilities_equity is None:
        return CheckResult(name, None, "total do ativo or total do passivo e PL not found")
    detail = f"{_fmt(assets)} vs {_fmt(liabilities_equity)}"
    return CheckResult(name, assets == liabilities_equity, detail)


def check_subtotals(statement: FinancialStatement) -> list[CheckResult]:
    """Every total line must equal the sum of the lines it aggregates.

    The statement has no explicit hierarchy, so we use a stack: walking the lines in
    order, each total consumes the most recent lines (going backwards) until their sum
    equals it, and then replaces them on the stack. Nested subtotals resolve naturally:
    'Total do ativo não circulante' consumes the subtotals that were already pushed.
    If no run of preceding lines adds up, the total does not close.
    """
    stack: list[tuple[str, Decimal]] = []
    results = []
    for line in statement.lines:
        if not line.is_total:
            stack.append((line.account_name, line.value))
            continue
        running = Decimal(0)
        matched_at = None
        for k in range(len(stack) - 1, -1, -1):
            running += stack[k][1]
            if running == line.value:
                matched_at = k
                break
        if matched_at is None:
            results.append(CheckResult(
                f"Subtotal: {line.account_name}", False,
                f"no run of preceding lines adds up to {_fmt(line.value)}"))
        else:
            parts = len(stack) - matched_at
            del stack[matched_at:]
            results.append(CheckResult(
                f"Subtotal: {line.account_name}", True,
                f"{_fmt(line.value)} = sum of {parts} line(s)"))
        stack.append((line.account_name, line.value))
    return results


def detect_scale(pages_text: str) -> CurrencyScale | None:
    """Read the scale from the statement header ('Em milhares de reais', 'R$ mil'...)."""
    text = normalize(pages_text)
    if re.search(r"milhoes de reais|em milhoes|r\$ ?milhoes|r\$ ?mm\b", text):
        return CurrencyScale.MILLION
    if re.search(r"milhares de reais|em milhares|r\$ ?mil\b|em r\$ mil", text):
        return CurrencyScale.THOUSAND
    if re.search(r"\bem reais\b|\(r\$\)", text):
        return CurrencyScale.UNIT
    return None


def check_scale(statement: FinancialStatement, pages_text: str | None) -> CheckResult:
    """The scale declared by the LLM must match the one printed in the PDF header."""
    name = "Scale matches PDF header"
    if pages_text is None:
        return CheckResult(name, None, "PDF text not available")
    printed = detect_scale(pages_text)
    if printed is None:
        return CheckResult(name, None, "scale not found in the statement pages")
    detail = f"extracted {statement.currency_scale.value}, PDF says {printed.value}"
    return CheckResult(name, printed == statement.currency_scale, detail)


# Expected sign of key lines, by CVM code: +1 must be positive, -1 must be negative.
# Results (gross profit, net income...) can legitimately be either, so they are not here.
EXPECTED_SIGNS: dict[StatementType, dict[str, int]] = {
    StatementType.BALANCE_SHEET: {"1": 1, "1.01": 1, "1.02": 1},  # asset totals
    StatementType.INCOME_STATEMENT: {"3.01": 1, "3.02": -1},  # revenue, cost of sales
}


def check_signs(statement: FinancialStatement) -> CheckResult:
    """Key lines must have the expected sign (e.g. revenue > 0, cost of sales < 0).

    In the income statement this catches the most common reading error: a value in
    parentheses copied as positive (or the opposite).
    """
    name = "Key lines have the expected sign"
    expected = EXPECTED_SIGNS.get(statement.statement_type, {})
    found = {code: _find_key_total(statement, code) for code in expected}
    if all(value is None for value in found.values()):
        return CheckResult(name, None, "no key lines found")
    wrong = [code for code, value in found.items()
             if value is not None and value * expected[code] <= 0]
    return CheckResult(name, not wrong, f"wrong sign: {wrong}" if wrong else "ok")


def run_checks(statement: FinancialStatement, pages_text: str | None = None) -> list[CheckResult]:
    """Run every check that applies to the statement type.

    `pages_text` (the statement pages) enables the scale check.
    """
    results = []
    if statement.statement_type == StatementType.BALANCE_SHEET:
        results.append(check_balance_equation(statement))
    results += [*check_subtotals(statement), check_scale(statement, pages_text),
                check_signs(statement)]
    for r in results:
        if r.passed is False:
            logger.warning("Integrity check failed - %s: %s", r.name, r.detail)
    return results


def summarize_checks(results: list[CheckResult]) -> dict:
    """Counts of passed / failed / inconclusive checks."""
    return {
        "checks_passed": sum(r.passed is True for r in results),
        "checks_failed": sum(r.passed is False for r in results),
        "checks_inconclusive": sum(r.passed is None for r in results),
    }


def statement_pages_text(statement: FinancialStatement, pdf_path: Path) -> str | None:
    """Text of the PDF pages holding the statement (None if the PDF is not available)."""
    if not pdf_path.exists():
        logger.warning("PDF %s not found; scale check will be inconclusive", pdf_path)
        return None
    pages = read_pages(pdf_path)
    return "\n".join(pages[i] for i in find_statement_pages(pages, statement.statement_type))


def check_file(json_path: Path) -> list[CheckResult]:
    """Run all checks on a statement saved by the extractor."""
    statement = load_statement(json_path)
    return run_checks(statement, statement_pages_text(statement, source_pdf_for(json_path)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Accounting checks on extracted statements.")
    parser.add_argument("extracted_json", type=Path, nargs="+")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    for json_path in args.extracted_json:
        results = check_file(json_path)
        print(f"\n== {json_path.name}: {summarize_checks(results)}")
        for r in results:
            status = {True: "ok  ", False: "FAIL", None: "n/a "}[r.passed]
            print(f"  [{status}] {r.name}: {r.detail}")


if __name__ == "__main__":
    main()
