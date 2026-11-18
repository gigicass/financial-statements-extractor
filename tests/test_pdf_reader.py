from src.extraction.pdf_reader import find_statement_pages, normalize
from src.extraction.schemas import StatementType

TOC_PAGE = "DEMONSTRAÇÕES FINANCEIRAS\nBalanço Patrimonial ..... 13\nDemonstração do Resultado ..... 15"
BALANCE_ASSETS = "Balanço Patrimonial em 31 de dezembro de 2025\nAtivo Circulante\nTOTAL DO ATIVO\n42.645.030"
BALANCE_LIABILITIES = "Balanço Patrimonial em 31 de dezembro de 2025\nPassivo\nTotal do Passivo\n10.000"
NOTES_PAGE = (
    "NOTAS EXPLICATIVAS ÀS DEMONSTRAÇÕES FINANCEIRAS EM 31 DE DEZEMBRO DE 2025\n"
    "O balanço patrimonial ... total do ativo ..."
)


def test_normalize_strips_accents_case_and_whitespace():
    assert normalize("  Balanço   PATRIMONIAL\n Líquido ") == "balanco patrimonial liquido"


def test_find_balance_sheet_pages_ignores_toc_and_notes():
    pages = [TOC_PAGE, BALANCE_ASSETS, BALANCE_LIABILITIES, NOTES_PAGE]
    assert find_statement_pages(pages, StatementType.BALANCE_SHEET) == [1, 2]


def test_find_statement_pages_returns_empty_when_absent():
    assert find_statement_pages([TOC_PAGE], StatementType.CASH_FLOW) == []
