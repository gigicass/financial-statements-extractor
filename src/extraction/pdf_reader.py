"""Read PDF text page by page and locate the pages of each financial statement."""

import logging
import unicodedata
from pathlib import Path

import pymupdf

from src.extraction.schemas import StatementType

logger = logging.getLogger(__name__)

# A statement starts on a page that contains the title AND at least one marker (a line
# that only appears inside the statement itself, which rules out the table of contents).
# All strings are normalized (lowercase, no accents) - see `normalize`.
# "excluded": a page containing any of these is never part of the statement - e.g. the
# statement of comprehensive income (DRA) shares the DRE's title words and its first
# line ("Lucro líquido do exercício") and usually comes right after it.
STATEMENT_KEYWORDS: dict[StatementType, dict[str, list[str]]] = {
    StatementType.BALANCE_SHEET: {
        "titles": ["balanco patrimonial", "balancos patrimoniais"],
        "markers": ["total do ativo", "total do passivo"],
        "excluded": [],
    },
    StatementType.INCOME_STATEMENT: {
        "titles": ["demonstracao do resultado", "demonstracoes do resultado",
                   "demonstracao dos resultados", "demonstracoes dos resultados",
                   "demonstracoes de resultados", "demonstracao de resultado"],
        "markers": ["lucro bruto", "resultado bruto"],
        "excluded": ["resultado abrangente", "resultados abrangentes"],
    },
    StatementType.CASH_FLOW: {
        "titles": ["fluxo de caixa", "fluxos de caixa"],
        "markers": ["atividades operacionais", "atividades de investimento"],
        "excluded": [],
    },
}

# Some PDFs bundle an earnings release before the official statements (e.g. Assaí).
# Release tables are managerial (often in R$ millions, sometimes adjusted figures), so
# pages with these markers are never taken as the start of a statement.
EARNINGS_RELEASE_MARKERS = ["divulgacao de resultados", "earnings release"]


def read_pages(pdf_path: str | Path) -> list[str]:
    """Return the text of each page of the PDF (index 0 = page 1)."""
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    logger.info("Read %d pages from %s", len(pages), pdf_path)
    return pages


def normalize(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace, for robust keyword matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.lower().split())


def find_statement_pages(pages: list[str], statement_type: StatementType) -> list[int]:
    """Return the 0-based indexes of the pages that contain the given statement.

    Takes the FIRST page with title + marker, plus the following pages that still have
    a marker (a statement split across pages often omits the title on the second page).
    Later matches are ignored: the statements come before the explanatory notes, and
    notes often repeat the title and totals (e.g. restatement tables).
    """
    keywords = STATEMENT_KEYWORDS[statement_type]
    texts = [normalize(page) for page in pages]

    def has_marker(text: str) -> bool:
        return (any(marker in text for marker in keywords["markers"])
                and not any(excluded in text for excluded in keywords["excluded"]))

    start = next((i for i, text in enumerate(texts)
                  if has_marker(text) and any(t in text for t in keywords["titles"])
                  and not any(r in text for r in EARNINGS_RELEASE_MARKERS)), None)
    if start is None:
        logger.info("%s not found", statement_type.value)
        return []

    found = [start]
    while found[-1] + 1 < len(texts) and has_marker(texts[found[-1] + 1]):
        found.append(found[-1] + 1)
    logger.info("%s found on pages %s", statement_type.value, [i + 1 for i in found])
    return found
