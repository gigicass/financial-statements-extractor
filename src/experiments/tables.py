"""Results tables built from results/*.csv (no plotting dependencies).

Used by the report (README tables) and by the Streamlit app.
"""

from pathlib import Path

import pandas as pd

from src.experiments.run_experiment import RESULTS_DIR, STUDY_COMPANIES

INJECTION_CSV = RESULTS_DIR / "error_injection.csv"
INJECTION_SOURCE = Path("data/processed/runs/models/v1/claude-opus-5-5-medium/r1")

MODEL_NAMES = {"claude-opus-5-5": "Opus 5.5", "claude-sonnet-5-5": "Sonnet 5.5",
               "claude-haiku-5-5": "Haiku 5.5"}
MODEL_ORDER = list(MODEL_NAMES)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"{name}.csv")


def _model_label(df: pd.DataFrame) -> pd.Series:
    effort = df["effort"].where(df["effort"] != "medium", "")
    return df["model"].map(MODEL_NAMES) + effort.map(lambda e: f" ({e} effort)" if e else "")


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return df.sort_values(["model", "effort"], key=lambda s: s.map(order) if s.name == "model"
                          else s.map({"medium": 0, "low": 1}))


def balance_sheet_by_model() -> pd.DataFrame:
    """Main table: balance sheets of the study companies, prompt v2, 3 repeats."""
    d = _read("models")
    d = d[(d["prompt"] == "v2") & d["company"].isin(STUDY_COMPANIES)]
    rows = []
    for (model, effort), g in _ordered(d).groupby(["model", "effort"], sort=False):
        rows.append({
            "Model": _model_label(g).iloc[0],
            "Runs": len(g),
            "Accounts found": f"{g['lines_coverage_pct'].mean():.1f}%",
            "Values exact": f"{g['lines_exact_match_pct'].mean():.1f}%",
            "Values within rounding": f"{g['lines_within_rounding_pct'].mean():.1f}%",
            "Runs failing accounting checks": int((g["checks_failed"] > 0).sum()),
            "Cost per BP (US$)": f"{g['cost_usd'].mean():.4f}",
            "Time per BP (s)": f"{g['seconds'].mean():.0f}",
        })
    return pd.DataFrame(rows)


def strategies_with_feedback() -> pd.DataFrame:
    """Cost per balance sheet when failed runs are retried once with feedback."""
    d = _read("models")
    d = d[(d["prompt"] == "v2") & d["company"].isin(STUDY_COMPANIES)]
    f = _read("models_feedback")
    keys = ["model", "effort", "repeat", "company"]
    m = d.merge(f[keys + ["feedback_cost_usd", "after_checks_failed"]], on=keys, how="left")
    rows = []
    for (model, effort), g in _ordered(m).groupby(["model", "effort"], sort=False):
        final_failed = g["after_checks_failed"].fillna(g["checks_failed"])
        rows.append({
            "Strategy": _model_label(g).iloc[0] + " + feedback",
            "Retries": int(g["feedback_cost_usd"].notna().sum()),
            "BPs passing all checks at the end": f"{int((final_failed == 0).sum())}/{len(g)}",
            "Cost per BP incl. retries (US$)":
                f"{(g['cost_usd'] + g['feedback_cost_usd'].fillna(0)).mean():.4f}",
            "_cost": (g["cost_usd"] + g["feedback_cost_usd"].fillna(0)).mean(),
        })
    return pd.DataFrame(rows)


def prompt_versions() -> pd.DataFrame:
    """Prompt v1 vs v2, first repeat, medium effort."""
    d = _read("models")
    d = d[(d["repeat"] == 1) & (d["effort"] == "medium") & d["company"].isin(STUDY_COMPANIES)]
    rows = []
    for (prompt, model), g in d.sort_values("prompt").groupby(["prompt", "model"], sort=False):
        rows.append({
            "Prompt": prompt, "Model": MODEL_NAMES[model],
            "Accounts found": f"{g['lines_coverage_pct'].mean():.1f}%",
            "Values within rounding": f"{g['lines_within_rounding_pct'].mean():.1f}%",
            "Extra lines": int(g["extra_lines"].sum()),
            "Failed checks": int(g["checks_failed"].sum()),
        })
    out = pd.DataFrame(rows)
    out["_order"] = out["Model"].map({MODEL_NAMES[m]: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["Prompt", "_order"]).drop(columns="_order")


def income_statement_by_model() -> pd.DataFrame:
    """DRE key totals (6 companies), with the feedback retry for failed checks."""
    d = pd.concat([_read("income_statement"), _read("income_statement_individual")])
    f = _read("income_statement_feedback")
    keys = ["model", "effort", "repeat", "company"]
    m = d.merge(f[keys + ["after_checks_failed", "feedback_cost_usd"]], on=keys, how="left")
    rows = []
    for model, g in _ordered(m).groupby("model", sort=False):
        exact = (g["totals_exact_match_pct"] * g["totals_found"] / 100).round().sum()
        rows.append({
            "Model": MODEL_NAMES[model],
            "Key lines found": f"{int(g['totals_found'].sum())}/{int(g['totals_accounts'].sum())}",
            "Key lines exact": f"{int(exact)}/{int(g['totals_found'].sum())}",
            "Runs failing checks": int((g["checks_failed"] > 0).sum()),
            "...after feedback": int((g["after_checks_failed"].fillna(g["checks_failed"]) > 0)
                                     .sum()),
            "Cost per DRE (US$)": f"{g['cost_usd'].mean():.4f}",
        })
    return pd.DataFrame(rows)


def hard_cases() -> pd.DataFrame:
    """Americanas (millions, judicial recovery) and Assaí (release + statements)."""
    cases = [("hard_cases", "Americanas (consolidated, R$ millions)"),
             ("hard_cases_manual_pages", "Assaí (individual, official statement pages)"),
             ("hard_cases_individual", "Assaí (individual, earnings-release page)")]
    rows = []
    for name, label in cases:
        for _, r in _ordered(_read(name)).iterrows():
            rows.append({
                "Case": label, "Model": MODEL_NAMES[r["model"]],
                "Key totals exact": f"{int(r['totals_found'])}/{int(r['totals_accounts'])}",
                "Accounts found": f"{r['lines_coverage_pct']:.1f}%",
                "Values within rounding": f"{r['lines_within_rounding_pct']:.1f}%",
                "Failed checks": int(r["checks_failed"]),
            })
    return pd.DataFrame(rows)


def error_detection(refresh: bool) -> pd.DataFrame:
    """Planted errors detected by integrity checks vs the CVM comparison."""
    if refresh or not INJECTION_CSV.exists():
        from src.evaluation.cvm_loader import DEFAULT_CVM_ZIP
        from src.evaluation.error_injection import run_experiment

        paths = sorted(INJECTION_SOURCE.glob("*_balance_sheet_consolidated.json"))
        run_experiment(paths, DEFAULT_CVM_ZIP).to_csv(INJECTION_CSV, index=False)
    raw = pd.read_csv(INJECTION_CSV)
    labels = {"drop_line": "Line skipped", "swap_digits": "Two digits swapped",
              "flip_sign": "Sign flipped", "wrong_scale": "Wrong scale (MIL vs UNIDADE)",
              "swap_neighbor_values": "Values swapped between two accounts"}
    g = raw.groupby("error").agg(trials=("error", "size"),
                                 integrity=("integrity_detected", "sum"),
                                 cvm=("cvm_detected", "sum")).reindex(list(labels))
    return pd.DataFrame({
        "Planted error": [labels[e] for e in g.index],
        "Trials": g["trials"].to_numpy(),
        "Accounting checks (no ground truth)": [f"{100 * i / t:.0f}%" for i, t in
                                                zip(g["integrity"], g["trials"])],
        "CVM comparison": [f"{100 * c / t:.0f}%" for c, t in zip(g["cvm"], g["trials"])],
        "_integrity": (100 * g["integrity"] / g["trials"]).to_numpy(),
        "_cvm": (100 * g["cvm"] / g["trials"]).to_numpy(),
    })
