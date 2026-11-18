"""Experiment: feed failed accounting checks back to the model and ask for a correction.

For every run in results/<experiment>.csv whose integrity checks failed, we send the
original prompt again, plus the model's previous answer and the list of failed checks,
and ask for the corrected statement. The correction is evaluated like any other run.
Only failed runs are retried, so the extra cost is paid only when something is wrong.

This lives in experiments/ because it combines extraction (the LLM call) with
evaluation (the integrity checks), which are kept independent of each other.

Usage:
    uv run python -m src.experiments.feedback --experiment models
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import DEFAULT_CVM_ZIP
from src.evaluation.evaluate import evaluate_file
from src.evaluation.integrity import CheckResult, check_file
from src.experiments.run_experiment import RAW_DIR, RESULTS_DIR, run_dir
from src.extraction.extractor import SYSTEM_PROMPT, save_statement, statement_prompt
from src.extraction.llm_client import get_client
from src.extraction.schemas import FinancialStatement, Scope, StatementType

logger = logging.getLogger(__name__)

FEEDBACK_INSTRUCTIONS = """Your previous extraction of this statement is below, \
followed by the accounting checks it FAILED. A subtotal that does not add up usually \
means a line was skipped, duplicated, taken from the wrong column or period, given the \
wrong sign, or placed on the wrong side (assets vs liabilities). Re-read the pages, \
find the cause, and return the full corrected statement following the same rules. \
Do not force totals to match: copy every value exactly as printed."""


def build_feedback_prompt(original_prompt: str, previous: FinancialStatement,
                          failed: list[CheckResult]) -> str:
    """Original prompt + previous answer + failed checks + instructions to fix."""
    previous_json = json.dumps(previous.model_dump(mode="json"), ensure_ascii=False, indent=1)
    failed_text = "\n".join(f"- {c.name}: {c.detail}" for c in failed)
    return (f"{original_prompt}\n\n{FEEDBACK_INSTRUCTIONS}\n\n"
            f"<previous_extraction>\n{previous_json}\n</previous_extraction>\n\n"
            f"<failed_checks>\n{failed_text}\n</failed_checks>")


def retry_run(row: pd.Series, experiment: str, statement_type: StatementType,
              scope: Scope) -> dict:
    """Retry one failed run with feedback; return before/after metrics and costs."""
    company, prompt = row["company"], row["prompt"]
    name = f"{company}_DFP_2025_{statement_type.value}_{scope.value}"
    before_path = run_dir(experiment, prompt, row["model"], row["effort"],
                          int(row["repeat"])) / f"{name}.json"
    after_dir = run_dir(f"{experiment}_feedback", prompt, row["model"], row["effort"],
                        int(row["repeat"]))
    after_path = after_dir / f"{name}.json"

    if not after_path.exists():
        previous = FinancialStatement.model_validate(
            json.loads(before_path.read_text(encoding="utf-8"))["statement"])
        failed = [c for c in check_file(before_path) if c.passed is False]
        pdf = RAW_DIR / f"{company}_DFP_2025.pdf"
        original = statement_prompt(pdf, statement_type, scope, prompt)
        client = get_client(model=row["model"], effort=row["effort"])
        result = client.extract(SYSTEM_PROMPT, build_feedback_prompt(original, previous, failed),
                                FinancialStatement)
        save_statement(result, name, after_dir, source_pdf=pdf, prompt_version=prompt)

    after = evaluate_file(after_path, DEFAULT_CVM_ZIP, details=False)
    after_run = json.loads(after_path.read_text(encoding="utf-8"))["run"]
    # Line-level metrics only exist when there is a reviewed mapping (not for the DRE).
    keep = ["totals_within_rounding_pct", "lines_coverage_pct", "lines_within_rounding_pct",
            "extra_lines", "checks_failed"]
    return {
        "prompt": prompt, "model": row["model"], "effort": row["effort"],
        "repeat": int(row["repeat"]), "company": company,
        **{f"before_{k}": row.get(k) for k in keep if k in after},
        **{f"after_{k}": after[k] for k in keep if k in after},
        "first_cost_usd": row["cost_usd"],
        "feedback_cost_usd": float(after_run["cost_usd"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed runs with check feedback.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt", default="v2")
    parser.add_argument("--statement", type=StatementType, default=StatementType.BALANCE_SHEET)
    parser.add_argument("--scope", type=Scope, default=Scope.CONSOLIDATED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    runs = pd.read_csv(RESULTS_DIR / f"{args.experiment}.csv")
    failed = runs[(runs["prompt"] == args.prompt) & (runs["checks_failed"] > 0)]
    logger.info("%d runs with failed checks will be retried with feedback", len(failed))

    results = pd.DataFrame([retry_run(row, args.experiment, args.statement, args.scope)
                            for _, row in failed.iterrows()])
    out = RESULTS_DIR / f"{args.experiment}_feedback.csv"
    results.to_csv(out, index=False)
    logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
