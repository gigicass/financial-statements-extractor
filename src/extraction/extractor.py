"""Orchestrate PDF -> statement pages -> LLM -> validated FinancialStatement -> files.

Usage:
    uv run python -m src.extraction.extractor data/raw/WEG_DFP_2025.pdf \
        --statement balance_sheet --scope consolidated
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from src.extraction.llm_client import LLMClient, LLMResult, get_client
from src.extraction.pdf_reader import find_statement_pages, read_pages
from src.extraction.schemas import FinancialStatement, Scope, StatementType

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

SYSTEM_PROMPT = """You extract financial statements from Brazilian company filings \
(CVM/IFRS) into structured data. Accuracy matters more than anything else: these \
values will be checked line by line against the official CVM dataset."""

STATEMENT_NAMES = {
    StatementType.BALANCE_SHEET: "Balanço Patrimonial (balance sheet: assets, liabilities and equity)",
    StatementType.INCOME_STATEMENT: "Demonstração do Resultado (income statement)",
    StatementType.CASH_FLOW: "Demonstração dos Fluxos de Caixa (cash flow statement)",
}

SCOPE_COLUMNS = {
    Scope.CONSOLIDATED: "CONSOLIDADO (consolidated)",
    Scope.INDIVIDUAL: "CONTROLADORA (parent company only)",
}


PROMPT_INTRO = """Below is text extracted from the PDF pages that contain the \
{statement}. The text came from a PDF table, so each cell may be on its own line and \
long account names may be split across two lines."""

# Rules per prompt version. Kept side by side so experiments stay reproducible.
PROMPT_RULES = {
    # v1: first version (phases 1-3).
    "v1": """Extract the statement following these rules:
1. Use only the {scope} columns, and only the MOST RECENT period.
2. Return every account line, in the order printed, with the name exactly as printed \
(join names split across lines; do not translate or abbreviate).
3. Values: copy them in the scale printed in the header (do not convert). Numbers in \
parentheses or with a minus sign are negative. A dash ('-') means 0.
4. Skip section headers that have no value of their own.
5. Mark subtotal and total lines with is_total=true. Some subtotals are printed \
without a label: name them after the section they total (e.g. 'Ativo Circulante').
6. Do not invent lines or compute values that are not printed.""",
    # v2: makes the output layout an explicit contract (order of subtotals, sides of
    # the balance sheet), after model comparison showed v1 was ambiguous.
    "v2": """Extract the statement following these rules:
1. Use only the {scope} columns, and only the MOST RECENT period.
2. Return every account line with the name exactly as printed (join names split \
across lines; do not translate, abbreviate or rename - this applies to subtotals too).
3. Values: copy them in the scale printed in the header (do not convert). Numbers in \
parentheses or with a minus sign are negative. A dash ('-') means 0.
4. Skip section headers that have no value of their own.
5. Order: list the lines in the order they are printed, and place every subtotal or \
total line immediately AFTER the lines it adds up - where its value is printed - never \
before them as a heading. Mark subtotal and total lines with is_total=true.
6. Some subtotals are printed without a label: name them after the section heading \
they total (e.g. 'Ativo Circulante').
7. Assets (Ativo) and liabilities and equity (Passivo e Patrimônio Líquido) may be \
printed side by side on the same page, so their lines can be interleaved in the text. \
Keep every account on its own side: never move a line from assets to liabilities or \
the other way round.
8. Do not invent lines or compute values that are not printed.""",
}
DEFAULT_PROMPT_VERSION = "v2"


def build_user_prompt(pages_text: str, statement_type: StatementType, scope: Scope,
                      prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Build the instructions plus the raw page text sent to the LLM."""
    intro = PROMPT_INTRO.format(statement=STATEMENT_NAMES[statement_type])
    rules = PROMPT_RULES[prompt_version].format(scope=SCOPE_COLUMNS[scope])
    return f"{intro}\n\n{rules}\n\n<pages>\n{pages_text}\n</pages>"


def statement_prompt(pdf_path: Path, statement_type: StatementType, scope: Scope,
                     prompt_version: str = DEFAULT_PROMPT_VERSION,
                     page_numbers: list[int] | None = None) -> str:
    """Find the statement pages in the PDF and build the user prompt around them.

    `page_numbers` (1-based) overrides the automatic page finder.
    """
    pages = read_pages(pdf_path)
    if page_numbers:
        page_indexes = [n - 1 for n in page_numbers]
        logger.info("Using pages %s given by the user", page_numbers)
    else:
        page_indexes = find_statement_pages(pages, statement_type)
    if not page_indexes:
        raise ValueError(f"No {statement_type.value} pages found in {pdf_path}")

    pages_text = "\n\n".join(
        f"--- page {i + 1} ---\n{pages[i]}" for i in page_indexes
    )
    return build_user_prompt(pages_text, statement_type, scope, prompt_version)


def extract_statement(
    pdf_path: Path,
    statement_type: StatementType,
    scope: Scope,
    client: LLMClient,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    page_numbers: list[int] | None = None,
) -> LLMResult[FinancialStatement]:
    """Find the statement pages in the PDF and ask the LLM to structure them."""
    prompt = statement_prompt(pdf_path, statement_type, scope, prompt_version, page_numbers)
    return client.extract(SYSTEM_PROMPT, prompt, FinancialStatement)


def save_statement(result: LLMResult[FinancialStatement], name: str, out_dir: Path,
                   source_pdf: Path | None = None,
                   prompt_version: str | None = None) -> Path:
    """Save the statement as CSV (one row per line) and JSON (with run metadata)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    statement = result.output

    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["company", "statement_type", "scope", "reference_date",
                         "currency_scale", "account_name", "value", "is_total"])
        for line in statement.lines:
            writer.writerow([statement.company, statement.statement_type.value,
                             statement.scope.value, statement.reference_date.isoformat(),
                             statement.currency_scale.value, line.account_name,
                             str(line.value), line.is_total])

    run_info = {
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "seconds": round(result.seconds, 1),
        "cost_usd": str(result.cost_usd) if result.cost_usd is not None else None,
        "source_pdf": source_pdf.as_posix() if source_pdf else None,
        "prompt_version": prompt_version,
    }
    json_path = out_dir / f"{name}.json"
    json_path.write_text(
        json.dumps({"run": run_info, "statement": statement.model_dump(mode="json")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %s and %s", csv_path, json_path)
    return csv_path


def load_statement(json_path: Path) -> FinancialStatement:
    """Read back a statement saved by `save_statement`."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return FinancialStatement.model_validate(data["statement"])


def source_pdf_for(json_path: Path, raw_dir: Path = Path("data/raw")) -> Path:
    """PDF a saved statement came from: stored in the JSON, or derived from its name.

    Older runs did not store it; their name is '<pdf stem>_<statement>_<scope>'.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    stored = data.get("run", {}).get("source_pdf")
    if stored:
        return Path(stored)
    statement = FinancialStatement.model_validate(data["statement"])
    suffix = f"_{statement.statement_type.value}_{statement.scope.value}"
    return raw_dir / f"{Path(json_path).stem.removesuffix(suffix)}.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a financial statement from a PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--statement", type=StatementType, default=StatementType.BALANCE_SHEET)
    parser.add_argument("--scope", type=Scope, default=Scope.CONSOLIDATED)
    parser.add_argument("--prompt", choices=sorted(PROMPT_RULES), default=DEFAULT_PROMPT_VERSION)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = extract_statement(args.pdf, args.statement, args.scope, get_client(),
                               prompt_version=args.prompt)
    name = f"{args.pdf.stem}_{args.statement.value}_{args.scope.value}"
    save_statement(result, name, PROCESSED_DIR, source_pdf=args.pdf,
                   prompt_version=args.prompt)


if __name__ == "__main__":
    main()
