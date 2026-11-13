"""Side-by-side comparison of model outputs for one experiment.

This shows where models differ. It does not rank them, merge them, or pick a
"best" answer: agreement between models is not evidence of correctness, and
only evaluation against human-verified ground truth measures accuracy.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from ..inference.output_parser import CATEGORY_DISPLAY_ORDER, CATEGORY_LABELS, get_field, item_text
from ..storage.experiment_manager import Experiment, ExperimentManager, utc_now, write_json, write_text_exact
from .textdist import normalise, similarity

AGREEMENT_CAVEAT = ("Agreement between models is NOT evidence of correctness: models can share the same "
                    "misreading. Accuracy can only be established against human-verified ground truth.")

# Comparison rows -> (label, source category, field extractor)
_FIELD_ROWS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("medicine_names", "Medicine names", "medications", ()),
    ("medicine_strengths", "Medicine strengths", "medications", ("strength",)),
    ("dosage_instructions", "Dosage instructions", "medication_instructions", ()),
]


def _values(structured: dict[str, Any] | None, category: str, fields: tuple[str, ...] = ()) -> list[str]:
    items = ((structured or {}).get("structured_extraction") or {}).get(category) or []
    out = []
    for it in items:
        v = get_field(it, *fields) if fields else item_text(it)
        if v and v.strip():
            out.append(v.strip())
    return out


def _agreement(per_model: dict[str, list[str]]) -> dict[str, Any]:
    """Compare item lists across models (set comparison after relaxed normalisation)."""
    keyed = {m: sorted(normalise(v, "relaxed") for v in vals) for m, vals in per_model.items()}
    models = list(keyed)
    if len(models) < 2:
        return {"status": "single_model", "pairwise": {}}
    pairwise = {}
    for a, b in combinations(models, 2):
        sa, sb = set(keyed[a]), set(keyed[b])
        union = sa | sb
        pairwise[f"{a} vs {b}"] = {"identical": keyed[a] == keyed[b],
                                   "jaccard": (len(sa & sb) / len(union)) if union else 1.0,
                                   "only_in_first": sorted(sa - sb), "only_in_second": sorted(sb - sa)}
    if all(not v for v in keyed.values()):
        status = "all_empty"
    elif all(p["identical"] for p in pairwise.values()):
        status = "identical"
    else:
        status = "disagreement"
    return {"status": status, "pairwise": pairwise}


def build_comparison(manager: ExperimentManager, exp: Experiment) -> dict[str, Any]:
    runs = {k: manager.load_model_run(exp, k) for k in manager.model_keys(exp)}
    usable = {k: r for k, r in runs.items() if r["raw_text"] is not None}
    report: dict[str, Any] = {
        "experiment_id": exp.exp_id, "generated_at": utc_now(), "caveat": AGREEMENT_CAVEAT,
        "image_sha256": exp.record["input"]["original_sha256"], "prompt_sha256": exp.record["prompt"]["sha256"],
        "models": {k: {"completion_status": (r["metadata"] or {}).get("completion_status"),
                       "is_mock": (r["metadata"] or {}).get("is_mock", False),
                       "structured_parse": (((r["metadata"] or {}).get("output") or {}).get("parse_status") or {})
                       .get("method"),
                       "has_output": r["raw_text"] is not None} for k, r in runs.items()},
        "contains_mock_runs": any((r["metadata"] or {}).get("is_mock") for r in runs.values()),
        "sections": {},
    }
    # Verbatim transcription
    transcriptions = {k: (r["structured"] or {}).get("verbatim_transcription") for k, r in usable.items()}
    pairwise = {}
    for a, b in combinations([k for k, t in transcriptions.items() if isinstance(t, str)], 2):
        pairwise[f"{a} vs {b}"] = {
            "identical": transcriptions[a] == transcriptions[b],
            "similarity_verbatim": round(similarity(transcriptions[a], transcriptions[b]), 4),
            "similarity_relaxed": round(similarity(normalise(transcriptions[a], "relaxed"),
                                                   normalise(transcriptions[b], "relaxed")), 4)}
    located = [k for k, t in transcriptions.items() if isinstance(t, str)]
    report["sections"]["verbatim_transcription"] = {
        "label": "Verbatim transcription", "values": transcriptions, "pairwise": pairwise,
        "status": ("single_model" if len(located) < 2 else
                   "identical" if all(p["identical"] for p in pairwise.values()) else "disagreement"),
        "not_located_in": [k for k in usable if k not in located]}
    # Clinical categories
    for key in CATEGORY_DISPLAY_ORDER:
        if key in ("medications", "medication_instructions", "other_unclassified"):
            continue
        values = {k: _values(r["structured"], key) for k, r in usable.items()}
        report["sections"][key] = {"label": CATEGORY_LABELS[key], "values": values, **_agreement(values)}
    for row_key, label, category, fields in _FIELD_ROWS:
        values = {k: _values(r["structured"], category, fields) for k, r in usable.items()}
        report["sections"][row_key] = {"label": label, "values": values, **_agreement(values)}
    order = ["verbatim_transcription", "chief_complaints", "clinical_history", "conditions_diagnoses",
             "clinical_observations", "measurements_vital_signs", "medicine_names", "medicine_strengths",
             "dosage_instructions", "investigations_tests", "advice_counselling", "follow_up_referral"]
    report["sections"] = {k: report["sections"][k] for k in order}
    report["disagreements"] = [k for k, s in report["sections"].items() if s["status"] == "disagreement"]
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Model comparison: {report['experiment_id']}", "", f"> {report['caveat']}", ""]
    if report.get("contains_mock_runs"):
        lines += ["> **Contains MOCK runs** (software test doubles, not model outputs).", ""]
    lines += [f"- Image SHA-256: `{report['image_sha256']}`", f"- Prompt SHA-256: `{report['prompt_sha256']}`", "",
              "| model | status | structured parse |", "|---|---|---|"]
    for k, m in report["models"].items():
        lines.append(f"| {k}{' (MOCK)' if m['is_mock'] else ''} | {m['completion_status']} | {m['structured_parse']} |")
    lines += ["", "Sections where the models differ: "
              + (", ".join(report["sections"][k]["label"] for k in report["disagreements"]) or "none") + ".", ""]
    for key, sec in report["sections"].items():
        mark = {"disagreement": "DIFFERENT", "identical": "same text", "all_empty": "empty in all models",
                "single_model": "one model only"}.get(sec["status"], sec["status"])
        lines += [f"## {sec['label']} ({mark})", ""]
        for model, value in sec["values"].items():
            lines.append(f"**{model}**")
            lines.append("")
            if value is None:
                lines.append("_not located in this model's output_")
            elif isinstance(value, str):
                lines += ["```", value, "```"]
            elif not value:
                lines.append("_no items_")
            else:
                lines += [f"- {v}" for v in value]
            lines.append("")
        for pair, p in (sec.get("pairwise") or {}).items():
            if "similarity_verbatim" in p:
                lines.append(f"- {pair}: character similarity {p['similarity_verbatim']:.3f} "
                             f"(relaxed {p['similarity_relaxed']:.3f})")
            elif not p["identical"]:
                lines.append(f"- {pair}: only in first {p['only_in_first']}; only in second {p['only_in_second']}")
        lines.append("")
    return "\n".join(lines)


def write_comparison(manager: ExperimentManager, exp: Experiment) -> dict[str, Any]:
    report = build_comparison(manager, exp)
    out = exp.path / "comparison"
    write_json(out / "comparison_report.json", report)
    write_text_exact(out / "comparison_report.md", render_markdown(report))
    return report
