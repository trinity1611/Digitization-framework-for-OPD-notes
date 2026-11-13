"""Prescription VLM zero-shot baseline: research interface.

Start with:   streamlit run app/main.py
Everything runs locally; the interface makes no network requests of its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Prescription VLM baseline", layout="wide")

PAGES = Path(__file__).parent / "pages"
navigation = st.navigation({
    "Setup": [
        st.Page(PAGES / "hardware_status.py", title="1. Hardware and environment", default=True),
        st.Page(PAGES / "model_management.py", title="2. Model management"),
    ],
    "Experiments": [
        st.Page(PAGES / "new_experiment.py", title="3. New experiment"),
        st.Page(PAGES / "results.py", title="4. Results"),
        st.Page(PAGES / "comparison.py", title="5. Comparison"),
        st.Page(PAGES / "history.py", title="6. Experiment history"),
    ],
    "Ground truth": [
        st.Page(PAGES / "annotation.py", title="7. Annotation"),
        st.Page(PAGES / "evaluation.py", title="8. Evaluation"),
    ],
})
with st.sidebar:
    st.caption("Zero-shot baseline: original image + fixed master prompt, one model at a time. "
               "No OCR, retrieval, catalogue, model collaboration or automatic correction.")
navigation.run()
