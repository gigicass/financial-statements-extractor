"""Smoke test: the app runs on the versioned sample without errors (no API calls)."""

import pytest

pytest.importorskip("streamlit")  # installed with: uv sync --group app

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def run_app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    return app


def test_app_renders_without_exceptions():
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "Financial Statements Extractor"


@pytest.mark.parametrize("statement", ["balance_sheet", "income_statement"])
def test_inspect_every_extraction_in_the_sample(statement):
    # Widgets use format_func, so they are driven with the underlying values (not the
    # labels shown on screen), taken from the sample manifest.
    manifest = pd.read_csv(APP.parent.parent / "data" / "samples" / "manifest.csv")
    app = run_app()
    app.selectbox[0].set_value(statement).run()
    for row in manifest[manifest["statement"] == statement].itertuples():
        app.selectbox[1].set_value(row.company).run()
        app.selectbox[2].set_value(row.model).run()
        app.radio[0].set_value(row.variant).run()
        assert not app.exception, (statement, row.company, row.model, row.variant)
