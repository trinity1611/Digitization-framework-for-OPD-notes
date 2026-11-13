import copy

from rxbench.inference.output_parser import (CATEGORY_KEYS, count_markers, get_field, item_text, parse_output,
                                             split_thinking)

MARKDOWN = """## A. Document overview

- Document type: Handwritten prescription
- Language: English
- Pre-existing printed or pre-filled content present: Yes
- Any uncertain handwritten/printed distinction: None

## B. Verbatim transcription

- T. Azee 500mg  1-0-0 x 7 days
- Syp Tusq 1 tsp [UNCERTAIN: "1-1-1" / "TDS"]
[ILLEGIBLE]

## C. Structured clinical extraction

### Chief complaints
None identified.

### Medications
1. **Original medicine text:** T. Azee 500mg
   - **Strength:** 500mg
   - **Formulation:** T.
   - **Evidence level:** EXPLICIT
2. **Original medicine text:** Syp Tusq
   - **Strength:** Not written
   - **Evidence level:** UNCERTAIN

### Medication instructions
- Original instruction: 1-0-0 x 7 days
- Frequency: 1-0-0
- Duration: x 7 days
- Associated medication: T. Azee 500mg

### Advice / counselling
| Original text | Interpreted advice | Evidence level |
|---|---|---|
| oral fluids | Drink fluids | EXPLICIT |

### Follow-up / referral
- Original text: f/u in med OPD SOS
- Interpreted instruction: follow up

## D. Uncertainty register

- Visible handwritten fragment: 1-1-1
- Possible readings: "1-1-1", "TDS"

## E. Final validation

1. All identifiable handwritten regions were considered.
"""


def test_markdown_sections_are_located():
    out = parse_output(MARKDOWN)
    st = out["parse_status"]
    assert st["method"] == "markdown_sections" and st["structured_parse_successful"] and not st["json_valid"]
    assert st["missing_sections"] == []
    assert out["document_overview"]["Language"] == "English"
    assert out["verbatim_transcription"].startswith("- T. Azee 500mg  1-0-0 x 7 days")   # double space preserved
    assert out["verbatim_transcription"].endswith("[ILLEGIBLE]")
    ext = out["structured_extraction"]
    assert ext["chief_complaints"] == [] and out["category_notes"]["chief_complaints"] == "model stated no items"
    assert [item_text(m) for m in ext["medications"]] == ["T. Azee 500mg", "Syp Tusq"]
    assert get_field(ext["medications"][0], "strength") == "500mg"
    assert get_field(ext["medication_instructions"][0], "associated medication") == "T. Azee 500mg"
    assert ext["advice_counselling"] == [{"Original text": "oral fluids", "Interpreted advice": "Drink fluids",
                                          "Evidence level": "EXPLICIT"}]
    assert item_text(ext["follow_up_referral"][0]) == "f/u in med OPD SOS"
    assert len(out["uncertainty_register"]["items"]) == 1
    assert out["uncertainty_markers"]["in_transcription"] == {**{k: 0 for k in count_markers("")},
                                                             "illegible": 1, "uncertain": 1}
    assert "clinical_history" in st["missing_categories"]


def test_every_structured_value_is_a_substring_of_the_raw_output():
    """Parsing must not add, correct or complete content."""
    out = parse_output(MARKDOWN)
    for key in CATEGORY_KEYS:
        for item in out["structured_extraction"].get(key, []):
            for value in item.values():
                for line in value.splitlines():
                    assert line.strip() in MARKDOWN


def test_parsing_does_not_modify_the_raw_string():
    raw = copy.copy(MARKDOWN)
    parse_output(raw)
    assert raw == MARKDOWN


def test_valid_json_is_parsed_as_is():
    out = parse_output('{"verbatim_transcription": "T. Azee", "extra": [1, 2]}')
    assert out["parse_status"]["json_valid"] and out["parse_status"]["method"] == "json"
    assert out["json"] == {"verbatim_transcription": "T. Azee", "extra": [1, 2]}
    assert "document_overview" in out["parse_status"]["missing_sections"]
    fenced = parse_output('```json\n{"a": 1}\n```')
    assert fenced["parse_status"]["json_method"] == "fenced_block" and fenced["json"] == {"a": 1}


def test_invalid_json_is_not_repaired():
    out = parse_output('{"verbatim_transcription": "T. Azee",}')        # trailing comma
    assert not out["parse_status"]["json_valid"] and not out["parse_status"]["structured_parse_successful"]
    assert "json" not in out


def test_plain_text_is_reported_as_unparsed():
    out = parse_output("The prescription says to take one tablet daily.")
    assert out["parse_status"]["method"] == "none" and not out["parse_status"]["structured_parse_successful"]


def test_reasoning_block_is_separated_for_parsing_only():
    thinking, answer = split_thinking("<think>\nlet me look\n</think>\n\n" + MARKDOWN)
    assert thinking.startswith("<think>") and answer == MARKDOWN
    assert split_thinking(MARKDOWN) == (None, MARKDOWN)
    out = parse_output("<think>x</think>\n" + MARKDOWN)
    assert out["reasoning_block_present"] and out["parse_status"]["structured_parse_successful"]


def test_marker_counts():
    text = '[ILLEGIBLE] a [PARTIALLY LEGIBLE: "para..."] [CROSSED OUT: old] [OVERWRITTEN: x] [HANDWRITING UNCLEAR] [illegible]'
    assert count_markers(text) == {"illegible": 2, "partially_legible": 1, "uncertain": 0, "crossed_out": 1,
                                   "overwritten": 1, "handwriting_unclear": 1}


# Shape of a real MedGemma 1.5 4B response (2026-11-13): an unsolicited reasoning block that itself
# contains bold pseudo-headings, items separated only by a repeated first field, and "N/A" placeholders.
REAL_SHAPE = """<unused94>thought
The user wants a transcription.

**Transcription Task (Task A):**

1.  **T. Azee 500mg x 7 days:**

**Structured Extraction Task (Task B):**

**Uncertainty Register:**

*   `T.` - Could be "Tab."

**Final Validation:** Check all constraints.<unused95>## A. Document overview

- Document type: Prescription
- Language: English

## B. Verbatim transcription

- Tab. Azee 500mg x 7 days
- oral fluids

## C. Structured clinical extraction

### Chief complaints
- Original text: N/A
- Interpretation: N/A
- Evidence level: N/A

### Medications
- Original text: T. Azee 500mg x 7 days
- Strength: 500mg
- Formulation: N/A
- Evidence level: Contextually supported
- Original text: Syp. Tussq 1 tsp
- Strength: N/A
- Evidence level: Contextually supported

### Advice / counselling
- Original text: oral fluids
- Interpreted advice: Oral fluids

## D. Uncertainty register

- `T.` - Could be "Tab." or "Tablet."

## E. Final validation

1.  All identifiable handwritten regions were considered. (Yes)
"""


def test_real_medgemma_output_shape_with_unsolicited_reasoning_block():
    out = parse_output(REAL_SHAPE)
    st = out["parse_status"]
    assert out["reasoning_block_present"] and out["reasoning_block_chars"] > 100
    assert st["method"] == "markdown_sections" and st["missing_sections"] == []
    assert st["markdown_sections_found"] == ["document_overview", "verbatim_transcription", "structured_extraction",
                                             "uncertainty_register", "final_validation"]
    assert out["document_overview"] == {"Document type": "Prescription", "Language": "English"}
    assert out["verbatim_transcription"] == "- Tab. Azee 500mg x 7 days\n- oral fluids"
    ext = out["structured_extraction"]
    assert ext["chief_complaints"] == [] and out["category_notes"]["chief_complaints"] == "model stated no items"
    assert [item_text(m) for m in ext["medications"]] == ["T. Azee 500mg x 7 days", "Syp. Tussq 1 tsp"]
    assert get_field(ext["medications"][0], "strength") == "500mg"
    assert item_text(ext["advice_counselling"][0]) == "oral fluids"
    assert out["uncertainty_register"]["items"] == [{"text": '`T.` - Could be "Tab." or "Tablet."'}]
    assert "thought" not in out["verbatim_transcription"]
