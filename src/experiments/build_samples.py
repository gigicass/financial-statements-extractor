"""Build the small, versioned data sample used by the Streamlit app.

The full inputs (PDFs, the CVM zip, every run) stay local and out of Git. The app only
needs, for each company and statement: the extracted JSONs (one per model, plus the
corrected one when the feedback loop ran), the CVM rows of that company, and images and
text of the statement pages. That fits in a few MB under data/samples/.

Usage:
    uv run python -m src.experiments.build_samples
"""

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pymupdf

from src.evaluation.cvm_loader import COMPANIES, DEFAULT_CVM_ZIP, load_cvm_statement
from src.experiments.run_experiment import RUNS_DIR, find_pdf
from src.extraction.extractor import load_statement
from src.extraction.pdf_reader import find_statement_pages, read_pages
from src.extraction.schemas import Scope, StatementType

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path("data/samples")
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
PAGE_DPI = 110
JPEG_QUALITY = 70

# (company, statement, scope, experiment folder) of the runs shown in the app:
# prompt v2, medium effort, first repeat.
CASES = (
    [(c, StatementType.BALANCE_SHEET, Scope.CONSOLIDATED, "models")
     for c in ("WEG", "KLBN", "CEAB", "PGMN")]
    + [("AMER", StatementType.BALANCE_SHEET, Scope.CONSOLIDATED, "hard_cases"),
       ("ASAI", StatementType.BALANCE_SHEET, Scope.INDIVIDUAL, "hard_cases_manual_pages")]
    + [(c, StatementType.INCOME_STATEMENT, Scope.CONSOLIDATED, "income_statement")
       for c in ("WEG", "KLBN", "CEAB", "PGMN", "AMER")]
    + [("ASAI", StatementType.INCOME_STATEMENT, Scope.INDIVIDUAL, "income_statement_individual")]
)


def copy_extractions(company: str, statement: StatementType, scope: Scope,
                     experiment: str) -> list[dict]:
    """Copy the run JSONs (and feedback corrections) into the sample; return manifest rows."""
    name = f"{company}_DFP_2025_{statement.value}_{scope.value}.json"
    rows = []
    for model in MODELS:
        for variant, folder in (("first", experiment), ("feedback", f"{experiment}_feedback")):
            src = RUNS_DIR / folder / "v2" / f"{model}-medium" / "r1" / name
            if not src.exists():
                continue
            dst = SAMPLES_DIR / "extractions" / variant / model / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            rows.append({"company": company, "statement": statement.value,
                         "scope": scope.value, "model": model, "variant": variant,
                         "path": dst.relative_to(SAMPLES_DIR).as_posix()})
    return rows


def save_pages(company: str, statement: StatementType) -> list[str]:
    """Render the statement pages as JPEG and save their text (for the scale check)."""
    pdf = find_pdf(company)
    pages = read_pages(pdf)
    indexes = find_statement_pages(pages, statement)
    out = SAMPLES_DIR / "pages"
    out.mkdir(parents=True, exist_ok=True)
    images = []
    with pymupdf.open(pdf) as doc:
        for i in indexes:
            image = out / f"{company}_{statement.value}_p{i + 1}.jpg"
            doc[i].get_pixmap(dpi=PAGE_DPI).save(image, jpg_quality=JPEG_QUALITY)
            images.append(image.relative_to(SAMPLES_DIR).as_posix())
    (out / f"{company}_{statement.value}.txt").write_text(
        "\n".join(pages[i] for i in indexes), encoding="utf-8")
    return images


def cvm_rows(company: str, statement: StatementType, scope: Scope) -> pd.DataFrame:
    cvm = load_cvm_statement(DEFAULT_CVM_ZIP, COMPANIES[company], statement, scope)
    return cvm.assign(company=company, statement=statement.value, scope=scope.value)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    for sub in ("extractions", "pages"):
        shutil.rmtree(SAMPLES_DIR / sub, ignore_errors=True)

    manifest, cvm, pages = [], [], {}
    for company, statement, scope, experiment in CASES:
        manifest += copy_extractions(company, statement, scope, experiment)
        cvm.append(cvm_rows(company, statement, scope))
        pages[f"{company}_{statement.value}"] = save_pages(company, statement)
        # Check that every copied file still validates.
        for row in manifest[-len(MODELS):]:
            load_statement(SAMPLES_DIR / row["path"])

    pd.DataFrame(manifest).to_csv(SAMPLES_DIR / "manifest.csv", index=False)
    pd.concat(cvm).to_csv(SAMPLES_DIR / "cvm_subset.csv", index=False)
    (SAMPLES_DIR / "pages" / "index.json").write_text(json.dumps(pages, indent=1),
                                                      encoding="utf-8")
    size = sum(f.stat().st_size for f in SAMPLES_DIR.rglob("*") if f.is_file())
    print(f"{len(manifest)} extractions, {len(pages)} page sets, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
