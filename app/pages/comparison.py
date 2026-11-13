"""Screen 5: comparison. Differences are highlighted; agreement is not treated as correctness."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.evaluation.comparison import AGREEMENT_CAVEAT, render_markdown, write_comparison  # noqa: E402

STATUS_LABEL = {"disagreement": "MODELS DIFFER", "identical": "same text in all models",
                "all_empty": "empty in all models", "single_model": "only one model has output"}


def render() -> None:
    ctx = common.get_ctx()
    common.header("Comparison")
    st.warning(AGREEMENT_CAVEAT)
    exp = common.select_experiment(ctx, "comparison_exp")
    if exp is None:
        return
    report = write_comparison(ctx.manager, exp)
    keys = common.ordered_model_keys(ctx, [k for k, m in report["models"].items() if m["has_output"]])
    if report["contains_mock_runs"]:
        common.mock_banner(True)
    without = [common.model_name(ctx, k) for k, m in report["models"].items() if not m["has_output"]]
    if without:
        st.info("No output to compare for: " + ", ".join(without))
    if not keys:
        st.info("No model output is available for this experiment.")
        return

    differing = [report["sections"][k]["label"] for k in report["disagreements"]]
    st.write("**Sections where the models differ:** " + (", ".join(differing) if differing else "none"))
    only_differences = st.toggle("Show only sections where the models differ", value=False)

    for section_key, section in report["sections"].items():
        if only_differences and section["status"] != "disagreement":
            continue
        label = STATUS_LABEL.get(section["status"], section["status"])
        with st.container(border=True):
            if section["status"] == "disagreement":
                st.markdown(f"### {section['label']}  \n:red[**{label}**]")
            else:
                st.markdown(f"### {section['label']}  \n:gray[{label}]")
            columns = st.columns(len(keys))
            for column, key in zip(columns, keys):
                with column:
                    st.markdown(f"**{common.model_name(ctx, key)}**")
                    value = section["values"].get(key)
                    if value is None:
                        st.caption("not located in this model's output")
                    elif isinstance(value, str):
                        st.code(value, language=None)
                    elif not value:
                        st.caption("no items")
                    else:
                        for item in value:
                            st.markdown(f"- `{item}`")
            pairwise = section.get("pairwise") or {}
            if section["status"] == "disagreement" and pairwise:
                rows = []
                for pair, p in pairwise.items():
                    a, b = pair.split(" vs ")
                    names = f"{common.model_name(ctx, a)} vs {common.model_name(ctx, b)}"
                    if "similarity_verbatim" in p:
                        rows.append({"pair": names, "identical": p["identical"],
                                     "character similarity": p["similarity_verbatim"],
                                     "similarity ignoring case and punctuation": p["similarity_relaxed"]})
                    else:
                        rows.append({"pair": names, "identical": p["identical"],
                                     "only in first": "; ".join(p["only_in_first"]),
                                     "only in second": "; ".join(p["only_in_second"])})
                st.dataframe(rows, hide_index=True)

    st.download_button("Download comparison report (Markdown)", render_markdown(report),
                       file_name=f"{exp.exp_id}_comparison_report.md", mime="text/markdown")
    st.caption("Similarity figures describe how alike two model outputs are. They are not accuracy; accuracy needs "
               "human-verified ground truth (Evaluation page).")


render()
