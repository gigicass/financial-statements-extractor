"""Streamlit app: explore the results and inspect any extraction side by side with the PDF.

The app makes no LLM calls: it only reads the versioned sample in data/samples/ (built by
src/experiments/build_samples.py), the reviewed mappings and the results CSVs.

Run locally:
    uv run --group app streamlit run app/streamlit_app.py
"""

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # Streamlit runs the script from app/, the code lives in src/

from src.evaluation.integrity import run_checks  # noqa: E402
from src.evaluation.matching import load_mapping, mapping_path  # noqa: E402
from src.evaluation.metrics import evaluate_key_totals, evaluate_lines, summarize  # noqa: E402
from src.experiments import tables  # noqa: E402
from src.extraction.extractor import load_statement  # noqa: E402
from src.extraction.schemas import FinancialStatement  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
FIGURES = ROOT / "docs" / "figures"
REPO_URL = "https://github.com/gigicass/financial-statements-extractor"

COMPANY_NAMES = {"WEG": "WEG", "KLBN": "Klabin", "CEAB": "C&A", "PGMN": "Pague Menos",
                 "AMER": "Americanas (hard case)", "ASAI": "Assaí (hard case)"}
STATEMENT_NAMES = {"balance_sheet": "Balance sheet (BP)",
                   "income_statement": "Income statement (DRE)"}
VARIANT_NAMES = {"first": "First answer", "feedback": "After feedback retry"}

st.set_page_config(page_title="Financial Statements Extractor", layout="wide")


# ---------------------------------------------------------------- data


@st.cache_data
def load_manifest() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "manifest.csv")


@st.cache_data
def load_cvm(company: str, statement: str, scope: str) -> pd.DataFrame:
    cvm = pd.read_csv(SAMPLES / "cvm_subset.csv", dtype={"code": str, "value": str})
    cvm = cvm[(cvm["company"] == company) & (cvm["statement"] == statement)
              & (cvm["scope"] == scope)].copy()
    cvm["value"] = cvm["value"].map(Decimal)
    return cvm


def page_images(company: str, statement: str) -> list[Path]:
    return sorted(SAMPLES.glob(f"pages/{company}_{statement}_p*.jpg"),
                  key=lambda p: int(p.stem.rsplit("_p", 1)[1]))


def page_text(company: str, statement: str) -> str | None:
    path = SAMPLES / "pages" / f"{company}_{statement}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def fmt(value, multiplier: int = 1) -> str:
    """Value in the statement's scale, Brazilian thousands separator (26.910.845)."""
    if value is None or pd.isna(value):
        return ""
    scaled = Decimal(value) / multiplier
    text = f"{scaled:,.0f}" if scaled == scaled.to_integral_value() else f"{scaled:,.2f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def status_of(row: pd.Series) -> str:
    if pd.isna(row["extracted_value"]) or row["extracted_value"] is None:
        return "missing"
    if row["exact_match"]:
        return "exact"
    return "within rounding" if row["within_rounding"] else "WRONG"


STATUS_COLORS = {"exact": "", "within rounding": "background-color: #fdf3d8",
                 "WRONG": "background-color: #fbe0df", "missing": "background-color: #fbe0df"}


# ---------------------------------------------------------------- tabs


def results_tab() -> None:
    st.markdown(
        "Six Brazilian listed companies, fiscal year 2025. Each statement was extracted "
        "from the PDF by three Claude models and **checked twice**: by accounting "
        "integrity checks that need no ground truth, and against the official CVM data.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance sheets extracted", "48", help="4 companies x 3 models x 3 repeats + "
              "Opus at low effort")
    c2.metric("Values right (within rounding)", "100%", help="Every model, prompt v2")
    c3.metric("Failing runs fixed by feedback", "9 / 9")
    c4.metric("Cheapest model per balance sheet", "US$ 0.003", help="Haiku 5.5")

    st.subheader("Balance sheet by model")
    st.dataframe(tables.balance_sheet_by_model(), hide_index=True)
    st.subheader("Cost when failed runs are retried with feedback")
    st.image(str(FIGURES / "cost_per_balance_sheet.png"))
    st.dataframe(tables.strategies_with_feedback().drop(columns="_cost"), hide_index=True)
    st.subheader("Income statement: key lines")
    st.dataframe(tables.income_statement_by_model(), hide_index=True)
    st.subheader("Who catches planted errors")
    st.image(str(FIGURES / "error_detection.png"))
    st.subheader("Prompt v1 vs v2")
    st.dataframe(tables.prompt_versions(), hide_index=True)
    st.subheader("Hard cases")
    st.dataframe(tables.hard_cases(), hide_index=True)


def inspect_tab() -> None:
    manifest = load_manifest()
    c1, c2, c3, c4 = st.columns(4)
    statement_type = c1.selectbox("Statement", list(STATEMENT_NAMES),
                                  format_func=STATEMENT_NAMES.get)
    rows = manifest[manifest["statement"] == statement_type]
    company = c2.selectbox("Company", [c for c in COMPANY_NAMES if c in set(rows["company"])],
                           format_func=COMPANY_NAMES.get)
    rows = rows[rows["company"] == company]
    model = c3.selectbox("Model", list(tables.MODEL_NAMES), format_func=tables.MODEL_NAMES.get)
    rows = rows[rows["model"] == model]
    variant = c4.radio("Answer", list(rows["variant"]), format_func=VARIANT_NAMES.get,
                       horizontal=True)
    row = rows[rows["variant"] == variant].iloc[0]

    statement = load_statement(SAMPLES / row["path"])
    cvm = load_cvm(company, statement_type, row["scope"])
    checks = run_checks(statement, page_text(company, statement_type))
    multiplier = statement.value_multiplier()

    left, right = st.columns([1, 1])
    with left:
        st.caption(f"PDF pages ({statement.currency_scale.value} = values in "
                   f"{'thousands' if multiplier == 1000 else 'millions' if multiplier > 1000 else 'units'} "
                   "of reais). Scope: " + row["scope"])
        for image in page_images(company, statement_type):
            st.image(str(image))
    with right:
        show_checks(checks)
        show_comparison(statement, cvm, company)
        with st.expander(f"Extracted lines ({len(statement.lines)})"):
            st.dataframe(pd.DataFrame([{
                "Account": line.account_name, "Value": fmt(line.value),
                "Subtotal": "yes" if line.is_total else ""} for line in statement.lines]),
                hide_index=True, height=400)


def show_checks(checks) -> None:
    failed = [c for c in checks if c.passed is False]
    st.markdown(f"**Accounting checks (no ground truth):** {len(checks) - len(failed)} passed, "
                f"{len(failed)} failed")
    for check in checks:
        icon = {True: "✅", False: "❌", None: "➖"}[check.passed]
        if check.passed is not True or not check.name.startswith("Subtotal"):
            st.markdown(f"{icon} {check.name}: {check.detail}")
    if not failed:
        st.caption("All subtotals add up.")


def show_comparison(statement: FinancialStatement, cvm: pd.DataFrame, company: str) -> None:
    multiplier = statement.value_multiplier()
    totals = evaluate_key_totals(statement, cvm)
    key = summarize(totals)
    st.markdown(f"**Key totals vs CVM:** {key['found']}/{key['accounts']} found, "
                f"{key['within_rounding_pct']:.0f}% right (within rounding)")
    path = mapping_path(company, statement)
    if not path.exists():
        st.caption("Line-by-line comparison is only available for balance sheets "
                   "(reviewed mappings).")
        table = totals.assign(status=totals.apply(status_of, axis=1))
        table = pd.DataFrame({"CVM code": table["code"], "CVM account": table["cvm_name"],
                              "PDF line": table["extracted_name"],
                              "CVM": table["cvm_value"].map(lambda v: fmt(v, multiplier)),
                              "Extracted": table["extracted_value"].map(
                                  lambda v: fmt(v, multiplier)),
                              "Status": table["status"]})
    else:
        lines, extras = evaluate_lines(statement, load_mapping(path), cvm)
        summary = summarize(lines)
        st.markdown(f"**Line by line vs CVM:** {summary['found']}/{summary['accounts']} "
                    f"accounts found, {summary['exact_match_pct']:.1f}% exact, "
                    f"{summary['within_rounding_pct']:.1f}% within rounding, "
                    f"{len(extras)} extra line(s)")
        if extras:
            st.warning("Extracted lines with no counterpart in the reviewed mapping: "
                       + "; ".join(extras))
        table = pd.DataFrame({"Account (PDF)": lines["account"],
                              "CVM code(s)": lines["cvm_codes"],
                              "CVM": lines["cvm_value"].map(lambda v: fmt(v, multiplier)),
                              "Extracted": lines["extracted_value"].map(
                                  lambda v: fmt(v, multiplier)),
                              "Status": lines.apply(status_of, axis=1)})
    styled = table.style.map(lambda s: STATUS_COLORS.get(s, ""), subset=["Status"])
    st.dataframe(styled, hide_index=True, height=420)


def about_tab() -> None:
    st.markdown(f"""
**Pipeline:** PDF → page finder → LLM with structured output (Pydantic schema) →
accounting checks → (retry with feedback if a check fails) → comparison with CVM data.

- **Accounting checks** need no ground truth: assets = liabilities + equity, every subtotal
  adds up, the scale matches the PDF header, key lines have the expected sign.
- **CVM comparison** uses a PDF → CVM account mapping reviewed by an accountant; values
  are never used to pair accounts, so a wrong value always shows up as an error.
- **Within rounding**: the CVM form rounds each account separately, so sums of CVM accounts
  can differ from the PDF by one unit of the scale (R$ 1 thousand).

Code, methodology and limitations: [GitHub repository]({REPO_URL}).
""")


st.title("Financial Statements Extractor")
st.caption("LLM extraction of Brazilian financial statements, measured against official "
           "CVM data.")
tab_results, tab_inspect, tab_about = st.tabs(["Results", "Inspect an extraction",
                                               "How it works"])
with tab_results:
    results_tab()
with tab_inspect:
    inspect_tab()
with tab_about:
    about_tab()
