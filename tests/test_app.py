"""Smoke test: the dashboard renders and live scoring works, using Streamlit's test runner."""

import pytest

from fraud.config import PROJECT_ROOT

APP = PROJECT_ROOT / "app" / "streamlit_app.py"
pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "app" / "data" / "alerts.parquet").exists(),
                                reason="run make app-data first")


def test_dashboard_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Fraud alert review"
    assert len(at.metric) >= 5


def test_score_a_payment_form():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    submit = next(b for b in at.button if b.label == "Score payment")
    submit.click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Alert?" in labels
