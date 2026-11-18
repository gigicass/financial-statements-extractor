import csv
import json

from src.extraction.extractor import build_user_prompt, save_statement
from src.extraction.llm_client import LLMResult
from src.extraction.schemas import FinancialStatement, Scope, StatementType


def make_result() -> LLMResult[FinancialStatement]:
    statement = FinancialStatement.model_validate({
        "company": "WEG S.A.",
        "statement_type": "balance_sheet",
        "scope": "consolidated",
        "reference_date": "2025-12-31",
        "currency_scale": "MIL",
        "lines": [
            {"account_name": "Caixa e equivalentes de caixa", "value": "6296498"},
            {"account_name": "TOTAL DO ATIVO", "value": "42645030", "is_total": True},
        ],
    })
    return LLMResult(output=statement, model="claude-opus-5-5",
                     input_tokens=5000, output_tokens=2000, seconds=20.0)


def test_prompt_names_the_requested_scope_and_includes_pages():
    prompt = build_user_prompt("PAGE TEXT", StatementType.BALANCE_SHEET, Scope.INDIVIDUAL)
    assert "CONTROLADORA" in prompt
    assert "<pages>\nPAGE TEXT\n</pages>" in prompt


def test_cost_estimate_uses_model_prices():
    # 5000 * $4/M + 2000 * $20/M = 0.02 + 0.04
    assert str(make_result().cost_usd) == "0.06"


def test_save_statement_writes_csv_and_json(tmp_path):
    csv_path = save_statement(make_result(), "weg_bs", tmp_path)

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["value"] for r in rows] == ["6296498", "42645030"]
    assert rows[1]["is_total"] == "True"

    saved = json.loads((tmp_path / "weg_bs.json").read_text(encoding="utf-8"))
    assert saved["run"]["model"] == "claude-opus-5-5"
    assert saved["statement"]["currency_scale"] == "MIL"


def test_prompt_v1_is_kept_unchanged_for_reproducibility():
    prompt = build_user_prompt("PAGE TEXT", StatementType.BALANCE_SHEET, Scope.CONSOLIDATED,
                               prompt_version="v1")
    assert "5. Mark subtotal and total lines with is_total=true." in prompt
    assert "immediately AFTER" not in prompt


def test_prompt_v2_states_the_layout_contract():
    prompt = build_user_prompt("PAGE TEXT", StatementType.BALANCE_SHEET, Scope.CONSOLIDATED)
    assert "immediately AFTER the lines it adds up" in prompt
    assert "never move a line from assets to liabilities" in prompt


def test_feedback_prompt_contains_previous_answer_and_failed_checks():
    from src.evaluation.integrity import CheckResult
    from src.experiments.feedback import build_feedback_prompt

    failed = [CheckResult("Subtotal: Total do passivo", False, "no run adds up to 6.827.539")]
    prompt = build_feedback_prompt("ORIGINAL PROMPT", make_result().output, failed)
    assert prompt.startswith("ORIGINAL PROMPT")
    assert "<previous_extraction>" in prompt and "Caixa e equivalentes de caixa" in prompt
    assert "- Subtotal: Total do passivo: no run adds up to 6.827.539" in prompt
