from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.extraction.schemas import CurrencyScale, FinancialStatement, StatementLine


def make_statement(**overrides) -> FinancialStatement:
    data = {
        "company": "WEG S.A.",
        "statement_type": "balance_sheet",
        "scope": "consolidated",
        "reference_date": "2025-12-31",
        "currency_scale": "MIL",
        "lines": [{"account_name": "TOTAL DO ATIVO", "value": "42645030", "is_total": True}],
    }
    data.update(overrides)
    return FinancialStatement.model_validate(data)


def test_value_is_parsed_as_exact_decimal():
    line = StatementLine(account_name="Caixa", value="6296498")
    assert line.value == Decimal("6296498")


def test_thousand_scale_multiplier():
    statement = make_statement()
    assert statement.currency_scale == CurrencyScale.THOUSAND
    assert statement.value_multiplier() == 1000


def test_statement_requires_at_least_one_line():
    with pytest.raises(ValidationError):
        make_statement(lines=[])


def test_unknown_scale_is_rejected():
    with pytest.raises(ValidationError):
        make_statement(currency_scale="MILHOES")
