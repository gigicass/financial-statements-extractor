from src.evaluation.integrity import (
    check_balance_equation,
    check_scale,
    check_signs,
    check_subtotals,
    detect_scale,
)
from src.extraction.schemas import CurrencyScale, FinancialStatement


def make_statement(lines, scale="MIL") -> FinancialStatement:
    return FinancialStatement.model_validate({
        "company": "X", "statement_type": "balance_sheet", "scope": "consolidated",
        "reference_date": "2025-12-31", "currency_scale": scale,
        "lines": [{"account_name": n, "value": v, "is_total": t} for n, v, t in lines],
    })


# A small but complete balance sheet with nested subtotals (as in Klabin's PDF)
# and equity split into controlling + non-controlling interests (as in WEG's).
GOOD = [
    ("Caixa", "60", False),
    ("Clientes", "40", False),
    ("Total do ativo circulante", "100", True),
    ("Depósitos judiciais", "5", False),
    ("Tributos diferidos", "15", False),
    ("Realizável a longo prazo", "20", True),
    ("Imobilizado", "70", False),
    ("Intangível", "10", False),
    ("Investimentos, imobilizado e intangível", "80", True),
    ("Total do ativo não circulante", "100", True),
    ("Total do ativo", "200", True),
    ("Fornecedores", "50", False),
    ("Total do passivo circulante", "50", True),
    ("Empréstimos", "70", False),
    ("Total do passivo não circulante", "70", True),
    ("Total do passivo", "120", True),
    ("Capital social", "90", False),
    ("Ações em tesouraria", "-15", False),
    ("Acionistas controladores", "75", True),
    ("Não controladores", "5", False),
    ("Total do patrimônio líquido", "80", True),
    ("Total do passivo e patrimônio líquido", "200", True),
]


def replace(lines, name, value=None, drop=False):
    out = []
    for n, v, t in lines:
        if n == name:
            if drop:
                continue
            v = value
        out.append((n, v, t))
    return out


def test_good_statement_passes_everything():
    statement = make_statement(GOOD)
    assert check_balance_equation(statement).passed
    assert all(r.passed for r in check_subtotals(statement))
    assert check_signs(statement).passed


def test_swapped_digits_break_the_subtotal():
    statement = make_statement(replace(GOOD, "Clientes", "4"))
    failed = [r.name for r in check_subtotals(statement) if not r.passed]
    assert failed == ["Subtotal: Total do ativo circulante"]


def test_missing_line_breaks_the_subtotal():
    statement = make_statement(replace(GOOD, "Clientes", drop=True))
    assert not all(r.passed for r in check_subtotals(statement))


def test_lost_sign_breaks_the_equity_subtotal():
    statement = make_statement(replace(GOOD, "Ações em tesouraria", "15"))
    failed = [r.name for r in check_subtotals(statement) if not r.passed]
    assert "Subtotal: Acionistas controladores" in failed


def test_unbalanced_statement_fails_equation():
    statement = make_statement(replace(GOOD, "Total do passivo e patrimônio líquido", "201"))
    assert check_balance_equation(statement).passed is False


def test_detect_scale_from_header():
    assert detect_scale("Balanço Patrimonial\n(Em milhares de reais)") == CurrencyScale.THOUSAND
    assert detect_scale("Balanço Patrimonial\nEm Reais") == CurrencyScale.UNIT
    assert detect_scale("Balanço Patrimonial") is None


def test_scale_check_flags_wrong_scale():
    statement = make_statement(GOOD, scale="UNIDADE")
    assert check_scale(statement, "Em milhares de reais").passed is False
    assert check_scale(statement, None).passed is None


def test_error_injectors_change_the_statement_as_described():
    import random

    from src.evaluation.error_injection import drop_line, flip_sign, swap_neighbor_values

    statement = make_statement(GOOD)
    rng = random.Random(0)
    assert len(drop_line(statement, rng).lines) == len(statement.lines) - 1

    flipped = flip_sign(statement, rng)
    assert sum(a.value != b.value for a, b in zip(statement.lines, flipped.lines)) == 1

    swapped = swap_neighbor_values(statement, rng)
    # Compensating error: total of all lines unchanged, so subtotals still close.
    assert sum(line.value for line in swapped.lines) == sum(line.value for line in statement.lines)
    assert all(r.passed for r in check_subtotals(swapped))
