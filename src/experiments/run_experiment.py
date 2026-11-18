"""Run extraction experiments: model x effort x company x repeat, then evaluate each run.

Every run is saved under data/processed/runs/<experiment>/<prompt>/<model>-<effort>/r<k>/
and one results row per run is appended to results/<experiment>.csv (versioned in Git).
Runs that already exist are skipped, so re-running never pays twice for the same run.

Usage:
    uv run python -m src.experiments.run_experiment --name models \
        --models claude-sonnet-5-5 claude-haiku-5-5 --repeats 1
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.cvm_loader import DEFAULT_CVM_ZIP
from src.evaluation.evaluate import evaluate_file
from src.extraction.extractor import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_RULES,
    extract_statement,
    save_statement,
)
from src.extraction.llm_client import get_client
from src.extraction.schemas import Scope, StatementType

logger = logging.getLogger(__name__)

RUNS_DIR = Path("data/processed/runs")
RESULTS_DIR = Path("results")
RAW_DIR = Path("data/raw")
STUDY_COMPANIES = ["WEG", "KLBN", "CEAB", "PGMN"]


def run_dir(experiment: str, prompt: str, model: str, effort: str, repeat: int) -> Path:
    return RUNS_DIR / experiment / prompt / f"{model}-{effort}" / f"r{repeat}"


def find_pdf(company: str) -> Path:
    """Study PDFs live in data/raw/, hard cases in data/raw/hard_cases/."""
    for folder in (RAW_DIR, RAW_DIR / "hard_cases"):
        if (pdf := folder / f"{company}_DFP_2025.pdf").exists():
            return pdf
    raise FileNotFoundError(f"No PDF for {company} in {RAW_DIR}")


def run_one(experiment: str, prompt: str, model: str, effort: str, repeat: int,
            company: str, statement_type: StatementType, scope: Scope,
            page_numbers: list[int] | None = None) -> dict:
    """Extract (if not done yet) and evaluate one company; return its results row."""
    pdf = find_pdf(company)
    out_dir = run_dir(experiment, prompt, model, effort, repeat)
    name = f"{pdf.stem}_{statement_type.value}_{scope.value}"
    json_path = out_dir / f"{name}.json"
    row = {"experiment": experiment, "prompt": prompt, "model": model, "effort": effort,
           "repeat": repeat, "company": company}

    if json_path.exists():
        logger.info("Skipping existing run %s", json_path)
    else:
        try:
            client = get_client(model=model, effort=effort)
            result = extract_statement(pdf, statement_type, scope, client, prompt,
                                       page_numbers)
        except Exception as e:  # one failed run must not stop the experiment
            logger.error("Run failed (%s %s r%d %s): %s", model, effort, repeat, company, e)
            return {**row, "error": str(e)[:200]}
        save_statement(result, name, out_dir, source_pdf=pdf, prompt_version=prompt)

    run_info = json.loads(json_path.read_text(encoding="utf-8"))["run"]
    row.update({k: run_info.get(k) for k in ("input_tokens", "output_tokens", "seconds",
                                              "cost_usd")})
    row.update(evaluate_file(json_path, DEFAULT_CVM_ZIP, details=False))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction experiments.")
    parser.add_argument("--name", required=True, help="experiment name (folder and CSV)")
    parser.add_argument("--prompts", nargs="+", default=[DEFAULT_PROMPT_VERSION],
                        choices=sorted(PROMPT_RULES))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--efforts", nargs="+", default=["medium"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--companies", nargs="+", default=STUDY_COMPANIES)
    parser.add_argument("--statement", type=StatementType, default=StatementType.BALANCE_SHEET)
    parser.add_argument("--scope", type=Scope, default=Scope.CONSOLIDATED)
    parser.add_argument("--pages", type=int, nargs="+",
                        help="1-based pages to use instead of the automatic page finder")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rows = [run_one(args.name, prompt, model, effort, repeat, company, args.statement,
                    args.scope, args.pages)
            for prompt in args.prompts
            for model in args.models
            for effort in args.efforts
            for repeat in range(1, args.repeats + 1)
            for company in args.companies]

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.csv"
    results = pd.DataFrame(rows)
    if out.exists():  # keep rows of other models/efforts run earlier in this experiment
        previous = pd.read_csv(out)
        keys = ["prompt", "model", "effort", "repeat", "company"]
        merged = previous.merge(results[keys], on=keys, how="left", indicator=True)
        results = pd.concat([previous[(merged["_merge"] == "left_only").to_numpy()], results])
    results.sort_values(["prompt", "model", "effort", "repeat", "company"]).to_csv(
        out, index=False)
    logger.info("Results written to %s (%d rows)", out, len(results))


if __name__ == "__main__":
    main()
