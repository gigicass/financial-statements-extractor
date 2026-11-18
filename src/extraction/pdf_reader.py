"""Read PDF text page by page and locate the pages of each financial statement."""

import logging
import unicodedata
from pathlib import Path

import pymupdf

from src.extraction.schemas import StatementType

logger = logging.getLogger(__name__)

# A page belongs to a statement when it contains the title AND at least one marker.
# Requiring a marker (a line that only appears inside the statement itself) filters out
# the table of contents and the explanatory notes, which also mention the titles.
# All strings are normalized (lowercase, no accents) - see `normalize`.
STATEMENT_KEYWORDS: dict[StatementType, dict[str, list[str]]] = {
    StatementType.BALANCE_SHEET: {
        "titles": ["balanco patrimonial", "balancos patrimoniais"],
        "markers": ["total do ativo", "total do passivo"],
    },
    StatementType.INCOME_STATEMENT: {
        "titles": ["demonstracao do resultado", "demonstracoes do resultado"],
        "markers": ["lucro liquido do exercicio", "receita operacional liquida"],
    },
    StatementType.CASH_FLOW: {
        "titles": ["fluxo de caixa", "fluxos de caixa"],
        "markers": ["atividades operacionais", "atividades de investimento"],
    },
}

# Header of the explanatory notes pages; they quote statement lines but are not statements.
NOTES_HEADER = "notas explicativas as demonstracoes financeiras em"


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
    """Return the 0-based indexes of the pages that contain the given statement."""
    keywords = STATEMENT_KEYWORDS[statement_type]
    found = []
    for index, page_text in enumerate(pages):
        text = normalize(page_text)
        if NOTES_HEADER in text:
            continue
        has_title = any(title in text for title in keywords["titles"])
        has_marker = any(marker in text for marker in keywords["markers"])
        if has_title and has_marker:
            found.append(index)
    logger.info("%s found on pages %s", statement_type.value, [i + 1 for i in found])
    return found
