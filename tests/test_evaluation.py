from decimal import Decimal

import pandas as pd

from src.evaluation.cvm_loader import normalize_cvm_rows
from src.evaluation.matching import MappingRow, parse_codes
from src.evaluation.metrics import compare_values, evaluate_key_totals, evaluate_lines, summarize
from src.extraction.schemas import FinancialStatement


def cvm_row(code, name, value, versao="1", ordem="ÚLTIMO", escala="MIL", fixa="S"):
    return {"CD_CONTA": code, "DS_CONTA": name, "VL_CONTA": value, "VERSAO": versao,
            "ORDEM_EXERC": ordem, "ESCALA_MOEDA": escala, "ST_CONTA_FIXA": fixa}


def test_normalize_keeps_latest_version_current_period_and_converts_scale():
    raw = pd.DataFrame([
        cvm_row("1", "Ativo Total", "100.0000000000", versao="1"),
        cvm_row("1", "Ativo Total", "120.0000000000", versao="2"),
        cvm_row("1", "Ativo Total", "90.0000000000", versao="2", ordem="PENÚLTIMO"),
        cvm_row("1.01", "Ativo Circulante", "70.5000000000", versao="2", fixa="N"),
    ])
    result = normalize_cvm_rows(raw)
    assert result["code"].tolist() == ["1", "1.01"]
    assert result["value"].tolist() == [Decimal("120000"), Decimal("70500")]
    assert result["level"].tolist() == [1, 2]
    assert result["is_fixed"].tolist() == [True, False]


def test_compare_values_exact_and_errors():
    assert compare_values(Decimal("100"), Decimal("100"))["exact_match"] is True
    diff = compare_values(Decimal("90"), Decimal("100"))
    assert diff["abs_error"] == Decimal("10") and diff["pct_error"] == Decimal("10")


def test_compare_values_separates_rounding_from_real_errors():
    # Statement in thousands: a R$ 1 mil difference is rounding, R$ 2 mil is not.
    off_by_one = compare_values(Decimal("2362018000"), Decimal("2362017000"), Decimal(1000))
    assert not off_by_one["exact_match"] and off_by_one["within_rounding"]
    off_by_two = compare_values(Decimal("2362019000"), Decimal("2362017000"), Decimal(1000))
    assert not off_by_two["within_rounding"]


def test_company_is_inferred_from_file_name():
    from pathlib import Path

    import pytest

    from src.evaluation.evaluate import company_from_path

    assert company_from_path(Path("KLBN_DFP_2025_balance_sheet_consolidated.json")) == "KLBN"
    with pytest.raises(ValueError):
        company_from_path(Path("XYZ_DFP_2025.json"))


def test_key_totals_converts_scale_and_reports_missing():
    statement = FinancialStatement.model_validate({
        "company": "X", "statement_type": "balance_sheet", "scope": "consolidated",
        "reference_date": "2025-12-31", "currency_scale": "MIL",
        "lines": [
            {"account_name": "TOTAL DO ATIVO", "value": "100", "is_total": True},
            {"account_name": "Ativo Circulante", "value": "61", "is_total": True},
            # "Total do Passivo" (liabilities only) must NOT be taken as CVM code 2.
            {"account_name": "Total do Passivo", "value": "40", "is_total": True},
        ],
    })
    codes = ["1", "1.01", "1.02", "2", "2.01", "2.02", "2.03"]
    cvm = pd.DataFrame({"code": codes, "name": codes,
                        "value": [Decimal(v) * 1000 for v in (100, 60, 40, 100, 30, 10, 60)]})

    results = evaluate_key_totals(statement, cvm).set_index("code")
    assert results.loc["1", "exact_match"]
    assert not results.loc["1.01", "exact_match"]
    assert results.loc["2", "extracted_value"] is None

    summary = summarize(results)
    assert summary["found"] == 2 and summary["exact_match_pct"] == 50.0



def test_parse_codes_handles_sums_and_subtractions():
    assert parse_codes("2.01+2.02") == [(1, "2.01"), (1, "2.02")]
    assert parse_codes("2.03-2.03.09") == [(1, "2.03"), (-1, "2.03.09")]
    assert parse_codes("") == []


def test_evaluate_lines_uses_section_for_repeated_names():
    statement = FinancialStatement.model_validate({
        "company": "X", "statement_type": "balance_sheet", "scope": "consolidated",
        "reference_date": "2025-12-31", "currency_scale": "MIL",
        "lines": [
            {"account_name": "Aplicações financeiras", "value": "10"},
            {"account_name": "Linha inventada", "value": "1"},
            {"account_name": "Ativo Circulante", "value": "11", "is_total": True},
            {"account_name": "Aplicações financeiras", "value": "7"},  # wrong: CVM says 5
            {"account_name": "Ativo Não Circulante", "value": "5", "is_total": True},
        ],
    })
    cvm = pd.DataFrame({"code": ["1.01", "1.01.02", "1.02", "1.02.01", "1.02.09"],
                        "name": [""] * 5,
                        "value": [Decimal(v) * 1000 for v in (11, 10, 5, 3, 2)]})
    mapping = [
        MappingRow("1.01", "Aplicações financeiras", [(1, "1.01.02")]),
        MappingRow("1.01", "Ativo Circulante", [(1, "1.01")]),
        MappingRow("1.02", "Aplicações financeiras", [(1, "1.02.01"), (1, "1.02.09")]),
        MappingRow("1.02", "Conta que o modelo pulou", [(1, "1.02.09")]),
        MappingRow("1.02", "Ativo Não Circulante", [(1, "1.02")]),
        MappingRow("1.02", "Linha sem código", []),
    ]

    results, extras = evaluate_lines(statement, mapping, cvm)
    by_key = {(r.section, r.account): r for r in results.itertuples()}

    assert by_key[("1.01", "Aplicações financeiras")].exact_match
    wrong = by_key[("1.02", "Aplicações financeiras")]
    assert not wrong.exact_match and wrong.abs_error == Decimal(2000)
    assert by_key[("1.02", "Conta que o modelo pulou")].extracted_value is None
    assert ("1.02", "Linha sem código") not in by_key
    assert extras == ["Linha inventada"]


def test_unlabeled_subtotal_is_paired_by_position_when_names_differ():
    # The PDF prints this subtotal without a label; another model named it differently.
    statement = FinancialStatement.model_validate({
        "company": "X", "statement_type": "balance_sheet", "scope": "consolidated",
        "reference_date": "2025-12-31", "currency_scale": "MIL",
        "lines": [
            {"account_name": "Depósitos judiciais", "value": "3"},
            {"account_name": "Não circulante", "value": "3", "is_total": True},
            {"account_name": "Imobilizado", "value": "2"},
            {"account_name": "Ativo Não Circulante", "value": "5", "is_total": True},
        ],
    })
    cvm = pd.DataFrame({"code": ["1.02", "1.02.01", "1.02.03"], "name": [""] * 3,
                        "value": [Decimal(v) * 1000 for v in (5, 3, 2)]})
    mapping = [
        MappingRow("1.02", "Depósitos judiciais", [(1, "1.02.01")]),
        MappingRow("1.02", "Realizável a longo prazo", [(1, "1.02.01")], is_total=True),
        MappingRow("1.02", "Imobilizado", [(1, "1.02.03")]),
        MappingRow("1.02", "Ativo Não Circulante", [(1, "1.02")], is_total=True),
    ]
    results, extras = evaluate_lines(statement, mapping, cvm)
    assert results["exact_match"].all()
    assert extras == []


def test_key_total_uses_last_line_when_partial_subtotal_has_the_same_name():
    # Unlabeled partial subtotal (before 'held for sale') named like the section total.
    statement = FinancialStatement.model_validate({
        "company": "X", "statement_type": "balance_sheet", "scope": "individual",
        "reference_date": "2025-12-31", "currency_scale": "MILHAO",
        "lines": [
            {"account_name": "Caixa", "value": "17782"},
            {"account_name": "Ativo Circulante", "value": "17782", "is_total": True},
            {"account_name": "Ativos mantidos para venda", "value": "18"},
            {"account_name": "Total do ativo circulante", "value": "17800", "is_total": True},
        ],
    })
    from src.evaluation.matching import find_key_total_line

    assert find_key_total_line(statement, "1.01").value == Decimal("17800")


def test_grouped_rows_are_compared_by_their_sum():
    # Two PDF lines that the CVM reports as one account (e.g. Assaí's tax payables).
    statement = FinancialStatement.model_validate({
        "company": "X", "statement_type": "balance_sheet", "scope": "individual",
        "reference_date": "2025-12-31", "currency_scale": "MILHAO",
        "lines": [
            {"account_name": "Demais impostos a recolher", "value": "473"},
            {"account_name": "IR e CS a pagar", "value": "45"},
            {"account_name": "Total do passivo circulante", "value": "518", "is_total": True},
        ],
    })
    cvm = pd.DataFrame({"code": ["2.01", "2.01.03"], "name": ["", ""],
                        "value": [Decimal(518_000_000), Decimal(518_000_000)]})
    mapping = [
        MappingRow("2.01", "Demais impostos a recolher", [(1, "2.01.03")], group="taxes"),
        MappingRow("2.01", "IR e CS a pagar", [(1, "2.01.03")], group="taxes"),
        MappingRow("2.01", "Total do passivo circulante", [(1, "2.01")], is_total=True),
    ]
    results, _ = evaluate_lines(statement, mapping, cvm)
    assert len(results) == 2
    taxes = results[results["account"].str.contains("Demais")].iloc[0]
    assert taxes["account"] == "Demais impostos a recolher + IR e CS a pagar"
    assert taxes["exact_match"]


def test_missing_partial_subtotal_is_reported_missing_not_wrong():
    # Earnings-release layout: no partial subtotal before 'held for sale'.
    statement = FinancialStatement.model_validate({
        "company": "X", "statement_type": "balance_sheet", "scope": "individual",
        "reference_date": "2025-12-31", "currency_scale": "MILHAO",
        "lines": [
            {"account_name": "Caixa", "value": "17782"},
            {"account_name": "Ativos mantidos para venda", "value": "18"},
            {"account_name": "Ativo Circulante", "value": "17800", "is_total": True},
        ],
    })
    cvm = pd.DataFrame({"code": ["1.01", "1.01.01", "1.01.08.01"], "name": [""] * 3,
                        "value": [Decimal(v) * 1_000_000 for v in (17800, 17782, 18)]})
    mapping = [
        MappingRow("1.01", "Caixa", [(1, "1.01.01")]),
        MappingRow("1.01", "Circulante", [(1, "1.01"), (-1, "1.01.08.01")], is_total=True),
        MappingRow("1.01", "Ativos mantidos para venda", [(1, "1.01.08.01")]),
        MappingRow("1.01", "Total do ativo circulante", [(1, "1.01")], is_total=True),
    ]
    results, extras = evaluate_lines(statement, mapping, cvm)
    by_account = results.set_index("account")
    assert by_account.loc["Total do ativo circulante", "exact_match"]
    assert by_account.loc["Circulante", "extracted_value"] is None
    assert extras == []
