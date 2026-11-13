# Evaluation guide

No accuracy figure is produced without human-verified ground truth for the same image. A model's
output is never used as ground truth, and agreement between models is never used in its place.

## 1. Two different things

| | what it is | needs ground truth | where |
|---|---|---|---|
| **Comparison** | where the models' outputs differ from each other | no | Comparison screen, `comparison/comparison_report.*` |
| **Evaluation** | how far each output is from what a person verified on the page | yes | Evaluation screen, `evaluation/evaluation_report.*` |

Comparison is useful for finding pages worth annotating. It says nothing about correctness: models can
share a misreading.

## 2. About the supplied dataset

`dataset_english/dataset_english.csv` has an English description per image, but it is a sentence-level
**paraphrase** ("Azee tablet should be taken once in the morning for seven days"), not a transcription
of what is written ("T. Azee 500mg 1-0-0 x 7 days"), and its own README notes rows that may be
truncated. It cannot serve as a reference for CER, WER or verbatim medicine matching, and the
application does not use it. Verbatim ground truth has to be annotated.

## 3. Annotating

One JSON file per image, named by the image's SHA-256, in `annotations/`.

- **Interface:** the Annotation screen shows the image and three tables (regions, printed text,
  medicines). It deliberately does not show any model output.
- **By hand:** `python -m rxbench annotation-template EXP_0001` writes a blank file.

Set `verified: true` and `verified_by` only after a person has checked every field against the image,
ideally someone other than the annotator. Evaluation refuses unverified annotations unless a
provisional run is explicitly requested, and marks such results PROVISIONAL.

### Format (version 1.0)

```json
{
  "annotation_version": "1.0",
  "image_sha256": "fb03cb2e...64 hex characters",
  "image_filename": "0022.jpg",
  "annotator": "AB",
  "verified": true,
  "verified_by": "CD",
  "verified_at": "2026-11-20",
  "notes": "",
  "regions": [
    {
      "region_id": "R1",
      "reading_order": 1,
      "location": "Middle-left",
      "bbox": [0.08, 0.28, 0.85, 0.36],
      "origin": "handwritten",
      "text": "T. Azee 500mg 1-0-0 x 7 days",
      "legibility": "legible",
      "acceptable_readings": [],
      "crossed_out": false,
      "category": "medications",
      "in_printed_field": false,
      "printed_label": null
    }
  ],
  "printed_text": [
    {"text": "Rx", "kind": "heading", "also_handwritten": false}
  ],
  "medications": [
    {
      "name_verbatim": "Azee", "name_normalised": "azee", "text_verbatim": "T. Azee 500mg",
      "strength": "500mg", "formulation": "T.", "route": null, "dose": null,
      "frequency": "1-0-0", "duration": "x 7 days", "instruction_verbatim": "1-0-0 x 7 days",
      "name_legibility": "legible"
    }
  ],
  "categories": {}
}
```

| field | meaning |
|---|---|
| `regions[].origin` | `handwritten`, `printed` or `uncertain` (cannot tell) |
| `regions[].text` | verbatim, as written: original spelling, abbreviations, numbers. Empty only for an illegible region |
| `regions[].legibility` | `legible`, `partially_legible`, `illegible`, as judged by the human reader |
| `regions[].acceptable_readings` | other readings a careful reader accepts for this region |
| `regions[].crossed_out` | struck-through handwriting (left out of the reference transcription) |
| `regions[].category` | one of the eleven category keys below, or a list of them |
| `regions[].in_printed_field`, `printed_label` | handwriting written into a printed field, and that field's printed label |
| `regions[].location`, `bbox` | one of the nine location labels; optional normalised box `[x_min, y_min, x_max, y_max]` |
| `printed_text[]` | pre-existing printed or pre-filled strings that must not be transcribed; `kind` is `heading`, `label`, `prefilled`, `hospital_info`, `instruction`, `boilerplate`, `table_header` or `other`; set `also_handwritten` when the doctor also wrote the same word |
| `medications[]` | one entry per medicine: name exactly as written, an optional normalised spelling of the same name, the full medicine text, and each attribute as written (null when not written) |
| `categories` | optional explicit lists of items per category; when filled it replaces the lists derived from `regions` |

Category keys: `chief_complaints`, `clinical_history`, `conditions_diagnoses`, `clinical_observations`,
`measurements_vital_signs`, `medications`, `medication_instructions`, `investigations_tests`,
`advice_counselling`, `follow_up_referral`, `other_unclassified`.

`name_normalised` is the annotator's own spelling normalisation of the written name (for example case).
It is not a lookup in a medicine catalogue, and brand names are not mapped to generics.

## 4. Running an evaluation

```
python -m rxbench evaluate EXP_0001
python -m rxbench evaluate EXP_0001 --annotation path/to/file.json
python -m rxbench evaluate EXP_0001 --allow-unverified      # provisional
```

The annotation's `image_sha256` must equal the experiment's image hash. Results are written to
`EXP_0001/evaluation/evaluation_report.json` and `.md`. A truncated model output is evaluated as it is
and flagged as truncated.

## 5. Metrics

All metrics compare strings. No dictionary, catalogue or model is consulted.

### A. Transcription

```
CER = (Substitutions + Deletions + Insertions) / Number of reference characters
WER = (Substitutions + Deletions + Insertions) / Number of reference words
```

Reference: the handwritten, non-crossed-out, readable regions in reading order, one per line.
Hypothesis: section B of the model's output. Three variants are reported side by side:

| variant | treatment |
|---|---|
| verbatim | strings compared exactly as they are |
| whitespace-normalised | runs of whitespace collapsed |
| relaxed, markers stripped | case-folded, punctuation removed (`/ + - % :` and decimal points kept), and uncertainty markers reduced to plain text (`[UNCERTAIN: "X" / "Y"]` → `X`; `[PARTIALLY LEGIBLE: "para..."]` → `para`; `[ILLEGIBLE]`, `[CROSSED OUT: ...]` removed) |

Separately:

- **Medicine names** and **dosage instructions**: item-matched CER. Items are paired one-to-one by
  similarity; an unmatched reference item counts as all its characters deleted, an unmatched predicted
  item as all its characters inserted.
- **Numerical values**: numbers as written (`500`, `6.8`, `130/80`, `1-0-1`) are extracted from reference
  and hypothesis; reported as a sequence error rate and as precision/recall over the bag of values.
- **Other clinical text**: item-matched CER over all categories except medications and their instructions.

If a model did not produce a locatable section B, its transcription is treated as empty (CER 100 %),
and `transcription_located` is false.

### B. Classification, per category

For each of the eleven categories independently: precision, recall and F1 at three matching levels
(exact string; identical after normalisation; similarity ≥ 0.8 after normalisation), plus *exact match*
(the predicted set of items equals the reference set) and *field-level accuracy* (share of reference
items reproduced exactly). Micro totals and macro F1 over categories with content are included.

### C. Medicines

Reference and predicted medicines are paired by name similarity. Reported separately:

- **exact medicine-name match**: the annotated name occurs, as whole tokens with identical characters,
  in the model's "Original medicine text";
- **normalised medicine-name match**: the same after case-folding and punctuation removal, also
  accepting the annotator's `name_normalised`;
- **strength, formulation, route, dose, frequency, duration accuracy**: exact, and normalised
  (`500 mg` equals `500mg`). Only medicines where the attribute was annotated are counted, and a
  medicine the model missed counts as wrong.

### D. Printed-text exclusion (separate from CER/WER)

1. *Printed headings transcribed*: share of annotated printed headings/labels that appear in the model's transcription.
2. *Pre-filled text extracted*: share of all annotated printed strings found in the transcription, and in the structured extraction.
3. *Handwriting in printed fields retained*: share of in-field handwritten regions found in the transcription (similarity ≥ 0.8).
4. *Confusion*: counts of printed text treated as handwritten and of in-field handwriting dropped, with the model's own statement about uncertain origin.
5. *Layout used without reproduction*: the in-field handwriting is kept, appears under the category its printed label implies, and the label itself is not reproduced.

### E. Uncertainty

Annotated regions are aligned, in reading order, with the lines of the model's transcription. A line
counts as *flagged* if it contains any of the prompt's uncertainty markers. Reported:

- illegible regions: correctly flagged, guessed without a flag, or omitted;
- legible regions incorrectly marked as uncertain;
- readable regions read wrongly: flagged versus **wrong and confident**;
- medicine names: correct or wrong, and whether a wrong name was at least flagged (marker, non-EXPLICIT
  evidence level, a "possible name", or an entry in the uncertainty register);
- the share of the model's errors that it had flagged.

This is what allows two models with the same CER to be told apart by how honestly they report doubt.

## 6. Reading the results

- Report the verbatim figures as the headline and say which variant any other figure is.
- One page is an anecdote. Aggregate over an annotated set, and report the number of pages and regions.
- Region-to-line alignment and item pairing are heuristics; check the `details` lists in the JSON report
  when a number looks surprising.
- Models are run at different effective image resolutions by design (see the protocol); say so when
  comparing them.
