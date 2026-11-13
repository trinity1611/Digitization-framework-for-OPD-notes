"""Interface smoke tests: every page renders without an exception, with and without stored experiments.
Uses MOCK adapters only (software test, not model inference)."""
from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from rxbench.config import PROJECT_ROOT  # noqa: E402
from rxbench.inference.generation import GenerationConfig  # noqa: E402
from rxbench.inference.protocol import run_protocol  # noqa: E402

PAGES = ["hardware_status.py", "model_management.py", "new_experiment.py", "results.py", "comparison.py",
         "history.py", "annotation.py", "evaluation.py"]


def _run(page: str) -> AppTest:
    st.cache_resource.clear()
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "pages" / page), default_timeout=120)
    at.run()
    return at


@pytest.fixture
def app_env(ctx, monkeypatch):
    monkeypatch.setenv("RXBENCH_ENABLE_MOCK", "1")
    return ctx


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_when_empty(app_env, page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_experiments(app_env, page, png_bytes, pdf_bytes):
    run_protocol(app_env, [("a.png", png_bytes), ("notes.pdf", pdf_bytes)], ["mock_alpha", "mock_beta"],
                 GenerationConfig(), pages="1")
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_results_page_labels_mock_runs_and_shows_raw_output(app_env, png_bytes):
    run_protocol(app_env, [("a.png", png_bytes)], ["mock_alpha", "mock_beta"], GenerationConfig())
    at = _run("results.py")
    assert any("MOCK RUN" in e.value for e in at.error)
    assert any("T. Mockamol" in c.value for c in at.code)


def test_comparison_page_shows_caveat_and_differences(app_env, png_bytes):
    run_protocol(app_env, [("a.png", png_bytes)], ["mock_alpha", "mock_beta"], GenerationConfig())
    at = _run("comparison.py")
    assert any("NOT evidence of correctness" in w.value for w in at.warning)
    assert any("MODELS DIFFER" in m.value for m in at.markdown)


def test_new_experiment_page_shows_independence_warning_and_prompt_check(app_env):
    at = _run("new_experiment.py")
    assert any("independently" in w.value for w in at.warning)
    assert any("exclusion rule is present" in s.value for s in at.success)
    assert at.button[0].disabled            # nothing can start without a document


def test_evaluation_page_shows_no_accuracy_without_ground_truth(app_env, png_bytes):
    run_protocol(app_env, [("a.png", png_bytes)], ["mock_alpha"], GenerationConfig())
    at = _run("evaluation.py")
    assert any("no annotation" in i.value for i in at.info)
    assert len(at.dataframe) == 0
