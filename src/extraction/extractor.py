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


def build_user_prompt(pages_text: str, statement_type: StatementType, scope: Scope) -> str:
    """Build the instructions plus the raw page text sent to the LLM."""
    return f"""Below is text extracted from the PDF pages that contain the \
{STATEMENT_NAMES[statement_type]}. The text came from a PDF table, so each cell may \
be on its own line and long account names may be split across two lines.

Extract the statement following these rules:
1. Use only the {SCOPE_COLUMNS[scope]} columns, and only the MOST RECENT period.
2. Return every account line, in the order printed, with the name exactly as printed \
(join names split across lines; do not translate or abbreviate).
3. Values: copy them in the scale printed in the header (do not convert). Numbers in \
parentheses or with a minus sign are negative. A dash ('-') means 0.
4. Skip section headers that have no value of their own.
5. Mark subtotal and total lines with is_total=true. Some subtotals are printed \
without a label: name them after the section they total (e.g. 'Ativo Circulante').
6. Do not invent lines or compute values that are not printed.

<pages>
{pages_text}
</pages>"""


def extract_statement(
    pdf_path: Path,
    statement_type: StatementType,
    scope: Scope,
    client: LLMClient,
) -> LLMResult[FinancialStatement]:
    """Find the statement pages in the PDF and ask the LLM to structure them."""
    pages = read_pages(pdf_path)
    page_indexes = find_statement_pages(pages, statement_type)
    if not page_indexes:
        raise ValueError(f"No {statement_type.value} pages found in {pdf_path}")

    pages_text = "\n\n".join(
        f"--- page {i + 1} ---\n{pages[i]}" for i in page_indexes
    )
    prompt = build_user_prompt(pages_text, statement_type, scope)
    return client.extract(SYSTEM_PROMPT, prompt, FinancialStatement)


def save_statement(result: LLMResult[FinancialStatement], name: str, out_dir: Path,
                   source_pdf: Path | None = None) -> Path:
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = extract_statement(args.pdf, args.statement, args.scope, get_client())
    name = f"{args.pdf.stem}_{args.statement.value}_{args.scope.value}"
    save_statement(result, name, PROCESSED_DIR, source_pdf=args.pdf)


if __name__ == "__main__":
    main()
