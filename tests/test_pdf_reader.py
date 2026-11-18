from src.extraction.pdf_reader import find_statement_pages, normalize
from src.extraction.schemas import StatementType

TOC_PAGE = "DEMONSTRAÇÕES FINANCEIRAS\nBalanço Patrimonial ..... 13\nDemonstração do Resultado ..... 15"
BALANCE_ASSETS = "Balanço Patrimonial em 31 de dezembro de 2025\nAtivo Circulante\nTOTAL DO ATIVO\n42.645.030"
BALANCE_LIABILITIES = "Balanço Patrimonial em 31 de dezembro de 2025\nPassivo\nTotal do Passivo\n10.000"
# Second page of a split statement, without the title (as in Klabin's PDF).
LIABILITIES_NO_TITLE = "PASSIVO\nCirculante\nFornecedores\nTotal do passivo\n10.000"
INCOME_STATEMENT = "Demonstração do Resultado\nReceita operacional líquida\nLucro líquido do exercício"
# A note that repeats the title and totals (e.g. a restatement table).
NOTES_PAGE = (
    "Notas explicativas da Administração às demonstrações financeiras\n"
    "2.1 Reapresentação - Balanço Patrimonial ... Total do passivo ..."
)


def test_normalize_strips_accents_case_and_whitespace():
    assert normalize("  Balanço   PATRIMONIAL\n Líquido ") == "balanco patrimonial liquido"


def test_finds_statement_pages_and_ignores_toc_and_later_notes():
    pages = [TOC_PAGE, BALANCE_ASSETS, BALANCE_LIABILITIES, INCOME_STATEMENT, NOTES_PAGE]
    assert find_statement_pages(pages, StatementType.BALANCE_SHEET) == [1, 2]


def test_includes_continuation_page_without_title():
    pages = [TOC_PAGE, BALANCE_ASSETS, LIABILITIES_NO_TITLE, INCOME_STATEMENT, NOTES_PAGE]
    assert find_statement_pages(pages, StatementType.BALANCE_SHEET) == [1, 2]


def test_find_statement_pages_returns_empty_when_absent():
    assert find_statement_pages([TOC_PAGE], StatementType.CASH_FLOW) == []


def test_skips_earnings_release_pages():
    release = "DIVULGAÇÃO DE RESULTADOS 4T25\nBalanço Patrimonial (R$ milhões)\nTotal do ativo\n47.825"
    pages = [TOC_PAGE, release, BALANCE_ASSETS, LIABILITIES_NO_TITLE, INCOME_STATEMENT]
    assert find_statement_pages(pages, StatementType.BALANCE_SHEET) == [2, 3]
