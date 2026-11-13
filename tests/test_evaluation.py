import copy

import pytest

from rxbench.evaluation import annotation as ann
from rxbench.evaluation import textdist
from rxbench.evaluation.cer import cer, item_matched_cer
from rxbench.evaluation.evaluate import AnnotationError, evaluate_experiment, evaluate_model_output
from rxbench.evaluation.field_metrics import category_metrics, match_counts
from rxbench.evaluation.medicine_metrics import medicine_metrics, name_core
from rxbench.evaluation.printed_text_metrics import printed_text_metrics
from rxbench.evaluation.uncertainty_metrics import uncertainty_metrics
from rxbench.evaluation.wer import numeric_values, wer
from rxbench.inference.generation import GenerationConfig
from rxbench.inference.output_parser import parse_output
from rxbench.inference.protocol import run_protocol

SHA = "a" * 64


def make_annotation(sha=SHA, verified=True):
    a = ann.empty_annotation(sha, "0022.jpg")
    a.update(annotator="t1", verified=verified, verified_by="t2" if verified else "")
    a["regions"] = [
        {"region_id": "R1", "reading_order": 1, "location": "Middle-left", "bbox": None, "origin": "handwritten",
         "text": "T. Azee 500mg 1-0-0 x 7 days", "legibility": "legible", "acceptable_readings": [],
         "crossed_out": False, "category": "medications", "in_printed_field": False, "printed_label": None},
        {"region_id": "R2", "reading_order": 2, "location": "Middle-left", "bbox": None, "origin": "handwritten",
         "text": "T. Pan 40mg 1-0-0 x 7days", "legibility": "partially_legible", "acceptable_readings": ["T. Lan 40mg 1-0-0 x 7days"],
         "crossed_out": False, "category": "medications", "in_printed_field": False, "printed_label": None},
        {"region_id": "R3", "reading_order": 3, "location": "Middle-centre", "bbox": None, "origin": "handwritten",
         "text": "", "legibility": "illegible", "acceptable_readings": [], "crossed_out": False,
         "category": "other_unclassified", "in_printed_field": False, "printed_label": None},
        {"region_id": "R4", "reading_order": 4, "location": "Bottom-left", "bbox": [0.1, 0.7, 0.6, 0.8], "origin": "handwritten",
         "text": "Acute gastritis", "legibility": "legible", "acceptable_readings": [], "crossed_out": False,
         "category": "conditions_diagnoses", "in_printed_field": True, "printed_label": "Diagnosis:"},
    ]
    a["printed_text"] = [{"text": "Diagnosis:", "kind": "label"}, {"text": "City Hospital", "kind": "hospital_info"},
                         {"text": "Rx", "kind": "heading", "also_handwritten": True}]
    a["medications"] = [
        {"name_verbatim": "Azee", "name_normalised": "azee", "text_verbatim": "T. Azee 500mg", "strength": "500mg",
         "formulation": "T.", "route": None, "dose": None, "frequency": "1-0-0", "duration": "x 7 days",
         "instruction_verbatim": "1-0-0 x 7 days", "name_legibility": "legible"},
        {"name_verbatim": "Pan", "name_normalised": "pan", "text_verbatim": "T. Pan 40mg", "strength": "40mg",
         "formulation": "T.", "route": None, "dose": None, "frequency": "1-0-0", "duration": "x 7days",
         "instruction_verbatim": "1-0-0 x 7days", "name_legibility": "partially_legible"},
    ]
    return a


MODEL_OUTPUT = """## A. Document overview
- Any uncertain handwritten/printed distinction: None

## B. Verbatim transcription
T. Azee 500mg 1-0-0 x 7 days
T. Ban 40 mg 1-0-0 x 7days
[ILLEGIBLE]
Diagnosis: Acute gastritis

## C. Structured clinical extraction

### Conditions / diagnoses
- Original text: Acute gastritis
- Evidence level: EXPLICIT

### Medications
- Original medicine text: T. Azee 500mg
- Strength: 500mg
- Formulation: T.
- Evidence level: EXPLICIT

- Original medicine text: T. Ban 40 mg
- Strength: 40 mg
- Formulation: T.
- Evidence level: EXPLICIT

### Medication instructions
- Original instruction: 1-0-0 x 7 days
- Frequency: 1-0-0
- Duration: x 7 days
- Associated medication: T. Azee 500mg

- Original instruction: 1-0-0 x 7days
- Frequency: 1-0-0
- Duration: 7 days
- Associated medication: T. Ban 40 mg

## D. Uncertainty register
None.
"""


# ---------------------------------------------------------------- CER / WER

def test_cer_formula():
    r = cer("kitten", "sitting")                 # 2 substitutions + 1 insertion over 6 reference characters
    assert (r["substitutions"], r["deletions"], r["insertions"]) == (2, 0, 1)
    assert r["value"] == pytest.approx(3 / 6) and r["reference_length"] == 6
    assert cer("abc", "abc")["value"] == 0.0
    assert cer("abcd", "")["value"] == 1.0 and cer("abcd", "")["deletions"] == 4
    assert cer("", "abc")["value"] is None and cer("", "abc")["insertions"] == 3


def test_wer_formula():
    r = wer("take one tablet daily", "take two tablets daily now")
    assert (r["substitutions"], r["deletions"], r["insertions"]) == (2, 0, 1)
    assert r["value"] == pytest.approx(3 / 4)


def test_verbatim_and_normalised_are_distinct():
    assert cer("Tab. Dolo 650", "tab dolo  650")["value"] > 0
    assert cer("Tab. Dolo 650", "tab dolo  650", "relaxed")["value"] == 0.0
    assert cer("a  b", "a b")["value"] > 0 and cer("a  b", "a b", "whitespace")["value"] == 0.0


def test_marker_stripping_is_explicit():
    hyp = 'Syp Tusq [UNCERTAIN: "TDS" / "BD"] [ILLEGIBLE]'
    assert textdist.strip_markers(hyp).split() == ["Syp", "Tusq", "TDS"]
    assert textdist.strip_markers('[PARTIALLY LEGIBLE: "para..."] x') == "para x"
    assert textdist.strip_markers("[CROSSED OUT: old] new").strip() == "new"
    assert cer("Syp Tusq TDS", hyp, "whitespace", strip_uncertainty_markers=True)["value"] == 0.0
    assert cer("Syp Tusq TDS", hyp)["uncertainty_markers_stripped"] is False


def test_pure_python_edit_counts_match_library():
    pairs = [("kitten", "sitting"), ("", "abc"), ("abc", ""), ("flaw", "lawn"), ("1-0-1 x 5 days", "1-1-1 x 5days"),
             (["a", "b", "c"], ["a", "x", "c", "d"])]
    for ref, hyp in pairs:
        s, d, i = textdist._edit_counts_python(ref, hyp)
        assert s + d + i == textdist.edit_counts(ref, hyp)["distance"]


def test_numeric_values():
    r = numeric_values("Azee 500mg 1-0-0 x 7 days BP 130/80", "Azee 500mg 1-0-1 x 7 days BP 130/80")
    assert r["reference_count"] == 4 and r["substitutions"] == 1
    assert r["bag_of_values"]["tp"] == 3 and r["sequence_error_rate"] == pytest.approx(0.25)


def test_item_matched_cer_counts_missing_and_extra_items():
    r = item_matched_cer(["Azee", "Dolo"], ["Azee", "Tusq", "Pan"])
    assert r["matched_pairs"] >= 1 and r["value"] > 0
    assert item_matched_cer(["Azee"], ["Azee"])["value"] == 0.0
    assert item_matched_cer(["Azee"], [])["value"] == 1.0


# ---------------------------------------------------------------- classification

def test_precision_recall_f1_levels():
    exact = match_counts(["Fever x 3 days", "Cough"], ["fever x 3 days", "Headache"], "exact")
    assert (exact["tp"], exact["fp"], exact["fn"]) == (0, 2, 2) and exact["f1"] == 0.0
    norm = match_counts(["Fever x 3 days", "Cough"], ["fever x 3 days", "Headache"], "normalised")
    assert (norm["tp"], norm["fp"], norm["fn"]) == (1, 1, 1) and norm["f1"] == pytest.approx(0.5)
    fuzzy = match_counts(["Fever x 3 days"], ["Fever x 3 day"], "fuzzy")
    assert fuzzy["tp"] == 1


def test_categories_are_evaluated_independently():
    ref = {"chief_complaints": ["Fever"], "conditions_diagnoses": ["Acute gastritis"]}
    pred = {"chief_complaints": ["Fever"], "conditions_diagnoses": ["Gastritis"], "advice_counselling": ["rest"]}
    m = category_metrics(ref, pred)
    assert m["per_category"]["chief_complaints"]["exact_match"] and m["per_category"]["chief_complaints"]["exact"]["f1"] == 1.0
    assert m["per_category"]["conditions_diagnoses"]["field_level_accuracy"] == 0.0
    assert m["per_category"]["advice_counselling"]["exact"]["fp"] == 1
    assert m["micro"]["exact"]["tp"] == 1 and m["categories_exact_match_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- medicines

def test_medicine_metrics_report_exact_and_normalised_separately():
    structured = parse_output(MODEL_OUTPUT)
    m = medicine_metrics(make_annotation()["medications"], structured)
    assert m["reference_medicines"] == 2 and m["paired"] == 2
    assert m["exact_medicine_name_match"] == 0.5 and m["normalised_medicine_name_match"] == 0.5   # "Ban" is not "Pan"
    acc = m["attribute_accuracy"]
    assert acc["strength"]["exact"] == 0.5 and acc["strength"]["normalised"] == 1.0   # "40 mg" vs "40mg"
    assert acc["formulation"]["exact"] == 1.0 and acc["frequency"]["exact"] == 1.0
    assert acc["duration"]["exact"] == 0.5 and acc["route"]["annotated"] == 0
    assert name_core("Tab. Dolo 650 mg") == "Dolo"


def test_missed_medicines_count_as_wrong():
    m = medicine_metrics(make_annotation()["medications"], parse_output("## B. Verbatim transcription\nnothing"))
    assert m["paired"] == 0 and m["missed_medicines"] == 2
    assert m["exact_medicine_name_match"] == 0.0 and m["attribute_accuracy"]["strength"]["exact"] == 0.0


# ---------------------------------------------------------------- printed text exclusion

def test_printed_text_metrics():
    structured = parse_output(MODEL_OUTPUT)
    m = printed_text_metrics(make_annotation(), structured["verbatim_transcription"], structured)
    assert m["1_printed_headings_transcribed"] == {"printed_headings_annotated": 1, "transcribed": 1, "rate": 1.0,
                                                   "items": ["Diagnosis:"]}
    assert m["2_prefilled_text_extracted"]["printed_items_annotated"] == 2          # "Rx" excluded: also handwritten
    assert m["2_prefilled_text_extracted"]["found_in_structured_extraction"] == 0
    assert m["3_handwriting_in_printed_fields_retained"]["rate"] == 1.0
    assert m["4_printed_handwritten_confusion"]["printed_treated_as_handwritten"] == 1
    assert m["5_layout_context_used_without_reproduction"]["rate"] == 0.0           # the label was reproduced
    clean = parse_output(MODEL_OUTPUT.replace("Diagnosis: Acute gastritis", "Acute gastritis"))
    m2 = printed_text_metrics(make_annotation(), clean["verbatim_transcription"], clean)
    assert m2["1_printed_headings_transcribed"]["rate"] == 0.0
    assert m2["5_layout_context_used_without_reproduction"]["rate"] == 1.0
    dropped = parse_output("## B. Verbatim transcription\nT. Azee 500mg 1-0-0 x 7 days\n")
    m3 = printed_text_metrics(make_annotation(), dropped["verbatim_transcription"], dropped)
    assert m3["3_handwriting_in_printed_fields_retained"]["dropped_regions"] == ["R4"]


# ---------------------------------------------------------------- uncertainty

def test_uncertainty_metrics():
    structured = parse_output(MODEL_OUTPUT)
    m = uncertainty_metrics(make_annotation(), structured["verbatim_transcription"], structured)
    assert m["illegible_regions"]["total"] == 1 and m["illegible_regions"]["correctly_flagged"] == 1
    assert m["readable_regions"]["total"] == 3
    assert m["readable_regions"]["wrong_and_confident"] >= 1          # "T. Ban 40 mg" written without any flag
    assert m["legible_regions_incorrectly_marked"]["marked_uncertain"] == 0
    assert m["medicine_names"]["wrong_and_confident"] == 1 and m["medicine_names"]["correct_and_confident"] == 1
    assert m["honesty_summary"]["share_of_errors_flagged"] == 0.0


def test_flagged_error_and_guessed_illegible_region():
    out = MODEL_OUTPUT.replace("T. Ban 40 mg 1-0-0 x 7days", 'T. [UNCERTAIN: "Ban" / "Pan"] 40 mg 1-0-0 x 7days') \
                      .replace("[ILLEGIBLE]\n", "Cap Omez 20\n")
    structured = parse_output(out)
    m = uncertainty_metrics(make_annotation(), structured["verbatim_transcription"], structured)
    assert m["readable_regions"]["wrong_but_flagged"] == 1
    assert m["illegible_regions"]["correctly_flagged"] == 0
    assert m["illegible_regions"]["guessed_without_flag"] + m["illegible_regions"]["omitted"] == 1


def test_acceptable_readings_count_as_correct():
    out = MODEL_OUTPUT.replace("T. Ban 40 mg 1-0-0 x 7days", "T. Lan 40mg 1-0-0 x 7days")
    structured = parse_output(out)
    m = uncertainty_metrics(make_annotation(), structured["verbatim_transcription"], structured)
    assert m["readable_regions"]["correct_and_confident"] >= 2


# ---------------------------------------------------------------- annotation and end-to-end

def test_annotation_validation():
    assert ann.validate(make_annotation()) == []
    bad = make_annotation()
    bad["regions"][0]["origin"] = "typed"
    bad["regions"][1]["category"] = "symptoms"
    bad["regions"][3]["bbox"] = [0.5, 0.5, 0.2, 2]
    bad["printed_text"][0]["kind"] = "logo"
    bad["image_sha256"] = "xyz"
    problems = "\n".join(ann.validate(bad))
    for fragment in ("origin", "unknown category 'symptoms'", "bbox", "kind", "image_sha256"):
        assert fragment in problems


def test_reference_transcription_and_categories():
    a = make_annotation()
    assert ann.reference_transcription(a).splitlines() == ["T. Azee 500mg 1-0-0 x 7 days", "T. Pan 40mg 1-0-0 x 7days",
                                                           "Acute gastritis"]
    cats = ann.reference_categories(a)
    assert cats["conditions_diagnoses"] == ["Acute gastritis"] and len(cats["medications"]) == 2
    a["regions"][0]["crossed_out"] = True
    assert "Azee" not in ann.reference_transcription(a)


def test_blank_annotation_contains_no_model_output():
    blank = ann.empty_annotation(SHA, "x.jpg")
    assert blank["verified"] is False and blank["regions"] == [] and blank["medications"] == []


def test_evaluate_model_output_end_to_end():
    m = evaluate_model_output(make_annotation(), parse_output(MODEL_OUTPUT))
    whole = m["transcription"]["whole_handwritten_text"]
    assert 0 < whole["cer_verbatim"]["value"] < 0.5
    assert whole["cer_relaxed_markers_stripped"]["value"] <= whole["cer_verbatim"]["value"]
    assert set(m) == {"transcription_located", "transcription", "classification", "medicines",
                      "printed_text_exclusion", "uncertainty"}
    assert m["classification"]["per_category"]["conditions_diagnoses"]["exact"]["f1"] == 1.0


def test_evaluation_requires_verified_ground_truth_for_the_same_image(ctx, png_bytes):
    import hashlib
    run_protocol(ctx, [("a.png", png_bytes)], ["mock_alpha", "mock_beta"], GenerationConfig())
    exp = ctx.manager.get("EXP_0001")
    sha = hashlib.sha256(png_bytes).hexdigest()
    with pytest.raises(AnnotationError, match="different image"):
        evaluate_experiment(ctx.manager, exp, make_annotation())
    with pytest.raises(AnnotationError, match="not marked as human-verified"):
        evaluate_experiment(ctx.manager, exp, make_annotation(sha, verified=False))
    provisional = evaluate_experiment(ctx.manager, exp, make_annotation(sha, verified=False), allow_unverified=True)
    assert provisional["ground_truth"]["provisional"]
    assert "PROVISIONAL" in (exp.path / "evaluation" / "evaluation_report.md").read_text(encoding="utf-8")
    report = evaluate_experiment(ctx.manager, exp, make_annotation(sha))
    assert set(report["models"]) == {"mock_alpha", "mock_beta"} and report["models"]["mock_alpha"]["is_mock"]
    assert report["models"]["mock_alpha"]["metrics"]["transcription"]["whole_handwritten_text"]["cer_verbatim"]["value"] > 0


def test_model_outputs_are_never_used_as_ground_truth(ctx, png_bytes):
    """The annotation template is blank; evaluation never reads another model's output as reference."""
    run_protocol(ctx, [("a.png", png_bytes)], ["mock_alpha", "mock_beta"], GenerationConfig())
    a = copy.deepcopy(make_annotation(ctx.manager.get("EXP_0001").record["input"]["original_sha256"]))
    report = evaluate_experiment(ctx.manager, ctx.manager.get("EXP_0001"), a)
    alpha = report["models"]["mock_alpha"]["metrics"]["medicines"]
    assert alpha["exact_medicine_name_match"] == 0.0          # mock medicines are not in the human annotation
