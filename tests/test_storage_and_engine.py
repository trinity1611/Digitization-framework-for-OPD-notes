"""Software tests of storage and the sequential engine using MOCK adapters (not model inference)."""
import hashlib
import io
import json
import logging
import zipfile

import pytest

from rxbench.evaluation.comparison import AGREEMENT_CAVEAT, write_comparison
from rxbench.inference.engine import InferenceEngine
from rxbench.inference.generation import GenerationConfig
from rxbench.inference.protocol import create_experiments, run_models, run_protocol
from rxbench.models import mock as mock_module
from rxbench.models.mock import mock_spec
from rxbench.storage import export
from rxbench.storage.experiment_manager import read_text_exact, write_text_exact
from tests.conftest import image_bytes

REQUIRED_METADATA = ["experiment_id", "completion_status", "input", "prompt", "model", "backend", "precision",
                     "quantisation", "image_processing", "generation", "timing", "tokens", "output", "memory",
                     "environment", "errors", "warnings", "is_mock"]
AWKWARD = "line one\r\nline two  \n\n\ttabbed nbsp → arrow\nno final newline"


# ---------------------------------------------------------------- storage

def test_experiment_preserves_original_bytes_and_prompt(ctx, prompt, png_bytes):
    exp = create_experiments(ctx, png_bytes, "scan.png", prompt)[0]
    assert exp.exp_id == "EXP_0001"
    assert exp.original_path.read_bytes() == png_bytes
    assert exp.record["input"]["original_sha256"] == hashlib.sha256(png_bytes).hexdigest()
    assert (exp.record["input"]["image"]["original_width"], exp.record["input"]["image"]["original_height"]) == (320, 240)
    assert (exp.path / "prompt" / "master_prompt_snapshot.txt").read_bytes() == ctx.prompt_path.read_bytes()
    assert (exp.path / "prompt" / "prompt_hash.txt").read_text().strip() == prompt.sha256
    assert exp.record["protocol"] == {"zero_shot": True, "ocr": False, "retrieval": False, "medicine_catalogue": False,
                                      "model_to_model_communication": False, "automatic_correction": False,
                                      "consensus_output": False}


def test_pdf_pages_become_separate_experiments_and_pdf_is_kept(ctx, prompt, pdf_bytes):
    exps = create_experiments(ctx, pdf_bytes, "notes.pdf", prompt, pages="1,3")
    assert [e.record["input"]["pdf"]["page_number"] for e in exps] == [1, 3]
    for e in exps:
        assert e.original_path.read_bytes() == pdf_bytes
        assert e.model_input_path.suffix == ".png"
        assert e.record["input"]["transformations"][0]["operation"] == "pdf_page_render"
    assert len(create_experiments(ctx, pdf_bytes, "notes.pdf", prompt)) == 3          # all pages by default


def test_tampered_input_or_prompt_is_detected(ctx, prompt, png_bytes):
    exp = create_experiments(ctx, png_bytes, "scan.png", prompt)[0]
    exp.model_input_path.write_bytes(image_bytes("PNG", text="changed"))
    with pytest.raises(ValueError):
        ctx.manager.load_prepared_image(exp)
    (exp.path / "prompt" / "master_prompt_snapshot.txt").write_text("shorter prompt")
    with pytest.raises(ValueError):
        ctx.manager.load_prompt(exp)


def test_text_is_written_and_read_back_exactly(tmp_path):
    path = tmp_path / "raw.txt"
    write_text_exact(path, AWKWARD)
    assert path.read_bytes() == AWKWARD.encode("utf-8")
    assert read_text_exact(path) == AWKWARD


def test_delete_only_removes_experiments(ctx, prompt, png_bytes):
    exp = create_experiments(ctx, png_bytes, "scan.png", prompt)[0]
    with pytest.raises(ValueError):
        ctx.manager.delete("../state")
    with pytest.raises(ValueError):
        ctx.manager.delete("_batches")
    ctx.manager.delete(exp.exp_id)
    assert ctx.manager.list_ids() == []


# ---------------------------------------------------------------- engine

def _run(ctx, files, models, gen=None, **kw):
    return run_protocol(ctx, files, models, gen or GenerationConfig(), **kw)


def test_all_models_get_the_same_image_and_prompt(ctx, png_bytes, jpeg_bytes):
    batch = _run(ctx, [("a.png", png_bytes), ("b.jpg", jpeg_bytes)], ["mock_alpha", "mock_beta"])
    assert batch["contains_mock_runs"] and batch["execution"] == "sequential"
    for exp in ctx.manager.list_experiments():
        metas = [ctx.manager.load_model_run(exp, k)["metadata"] for k in ("mock_alpha", "mock_beta")]
        assert len({m["input"]["image_sha256"] for m in metas}) == 1
        assert len({m["input"]["pixel_sha256_given_to_model"] for m in metas}) == 1
        assert metas[0]["input"]["pixel_sha256_given_to_model"] == metas[0]["input"]["pixel_sha256_expected"]
        assert {m["prompt"]["sha256_given_to_model"] for m in metas} == {exp.record["prompt"]["sha256"]}
        for m in metas:
            assert m["completion_status"] == "complete" and m["is_mock"] and "MOCK" in m["mock_notice"]
            assert not [k for k in REQUIRED_METADATA if k not in m]
            assert m["memory"]["inference"]["primary_peak_basis"] and m["memory"]["after_unload_check"] is not None
            assert m["timing"]["generation_seconds"] is not None and m["tokens"]["output"] > 0
            assert not any("PROTOCOL WARNING" in w for w in m["warnings"])


def test_raw_output_is_preserved_exactly_and_parsing_does_not_touch_it(ctx, png_bytes):
    ctx.cfg["models"]["mock_alpha"] = {"mock": {"output_text": AWKWARD}}
    _run(ctx, [("a.png", png_bytes)], ["mock_alpha"])
    exp = ctx.manager.get("EXP_0001")
    run = ctx.manager.load_model_run(exp, "mock_alpha")
    assert run["raw_text"] == AWKWARD
    assert (exp.path / "mock_alpha" / "raw_output.txt").read_bytes() == AWKWARD.encode("utf-8")
    assert run["metadata"]["output"]["raw_output_sha256"] == hashlib.sha256(AWKWARD.encode("utf-8")).hexdigest()
    assert run["metadata"]["output"]["parse_status"]["structured_parse_successful"] is False   # plain text
    assert run["structured"]["parse_status"]["method"] == "none"


def test_one_model_failing_does_not_stop_the_others(ctx, png_bytes):
    ctx.registry.register("mock_noload", mock_spec("mock_noload", "alpha", fail_on_load=True))
    ctx.registry.register("mock_nogen", mock_spec("mock_nogen", "alpha", fail_on_generate=True))
    batch = _run(ctx, [("a.png", png_bytes)], ["mock_noload", "mock_nogen", "mock_beta"])
    exp = ctx.manager.get("EXP_0001")
    status = {k: ctx.manager.load_model_run(exp, k)["metadata"]["completion_status"]
              for k in ("mock_noload", "mock_nogen", "mock_beta")}
    assert status == {"mock_noload": "failed_model_load", "mock_nogen": "failed_inference", "mock_beta": "complete"}
    assert ctx.manager.load_model_run(exp, "mock_noload")["raw_text"] is None
    assert ctx.manager.load_model_run(exp, "mock_noload")["metadata"]["errors"]
    assert [r["load_ok"] for r in batch["model_records"]] == [False, True, True]


def test_unavailable_real_model_is_reported_not_substituted(ctx, png_bytes, monkeypatch):
    monkeypatch.setattr(ctx.downloads, "local_status", lambda spec: {"status": "not_downloaded", "path": None})
    _run(ctx, [("a.png", png_bytes)], ["qwen3vl_32b", "mock_alpha"])
    exp = ctx.manager.get("EXP_0001")
    meta = ctx.manager.load_model_run(exp, "qwen3vl_32b")["metadata"]
    assert meta["completion_status"] == "model_not_available_locally" and not meta["is_mock"]
    assert meta["model"]["identifier"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert ctx.manager.load_model_run(exp, "mock_alpha")["metadata"]["completion_status"] == "complete"


def test_truncated_output_is_marked_incomplete_and_rerun_keeps_history(ctx, png_bytes):
    _run(ctx, [("a.png", png_bytes)], ["mock_alpha"], GenerationConfig(max_new_tokens=20))
    exp = ctx.manager.get("EXP_0001")
    first = ctx.manager.load_model_run(exp, "mock_alpha")
    assert first["metadata"]["completion_status"] == "incomplete_truncated"
    assert first["metadata"]["tokens"]["truncated"] and not first["metadata"]["output_complete"]
    assert first["raw_text"]                                          # truncated text is still kept
    run_models(ctx, [exp.exp_id], ["mock_alpha"], GenerationConfig(max_new_tokens=8192))
    second = ctx.manager.load_model_run(exp, "mock_alpha")
    assert second["metadata"]["completion_status"] == "complete"
    history = ctx.manager.history(exp, "mock_alpha")
    assert len(history) == 1
    old = exp.path / "mock_alpha" / "history" / history[0]
    assert read_text_exact(old / "raw_output.txt") == first["raw_text"]
    assert json.loads((old / "metadata.json").read_text(encoding="utf-8"))["completion_status"] == "incomplete_truncated"


def test_models_run_strictly_one_after_another_and_never_see_each_other(ctx, png_bytes, monkeypatch):
    events, seen_inputs = [], []
    original_generate = mock_module.MockAdapter.generate

    def spy(self, messages, gen):
        seen_inputs.append((self.spec.key, set(messages), messages["prompt"]))
        return original_generate(self, messages, gen)

    monkeypatch.setattr(mock_module.MockAdapter, "generate", spy)
    _run(ctx, [("a.png", png_bytes)], ["mock_alpha", "mock_beta"], progress=lambda e, i: events.append((e, i.get("model"))))
    lifecycle = [e for e in events if e[0] in ("model_loading", "inference_done", "model_done")]
    assert lifecycle == [("model_loading", "mock_alpha"), ("inference_done", "mock_alpha"), ("model_done", "mock_alpha"),
                         ("model_loading", "mock_beta"), ("inference_done", "mock_beta"), ("model_done", "mock_beta")]
    prompt_text = ctx.prompt_path.read_bytes().decode("utf-8")
    for _key, fields, given_prompt in seen_inputs:
        assert fields == {"image", "prompt"}            # nothing but the page and the prompt
        assert given_prompt == prompt_text              # identical, unshortened prompt for every model
        assert mock_module.MOCK_BANNER not in given_prompt


def test_skip_completed_resumes_without_repeating(ctx, png_bytes):
    _run(ctx, [("a.png", png_bytes)], ["mock_alpha"])
    batch = run_models(ctx, ["EXP_0001"], ["mock_alpha", "mock_beta"], GenerationConfig(), skip_completed=True)
    runs = {r["model_key"]: r["runs"] for r in batch["model_records"]}
    assert runs["mock_alpha"] == {"EXP_0001": "skipped_already_complete"} and runs["mock_beta"] == {"EXP_0001": "complete"}
    assert ctx.manager.history(ctx.manager.get("EXP_0001"), "mock_alpha") == []


def test_simultaneous_serving_must_be_explicit_and_is_not_silently_enabled(ctx):
    assert ctx.cfg["execution"]["allow_simultaneous"] is False
    ctx.cfg["execution"]["allow_simultaneous"] = True
    with pytest.raises(NotImplementedError):
        InferenceEngine(ctx.cfg, ctx.registry, ctx.manager, ctx.downloads)


def test_prompt_without_exclusion_rule_is_refused(ctx, png_bytes, tmp_path):
    bad = tmp_path / "prompt.txt"
    bad.write_text("Transcribe everything on the page.")
    ctx.prompt_path = bad
    with pytest.raises(ValueError):
        _run(ctx, [("a.png", png_bytes)], ["mock_alpha"])


# ---------------------------------------------------------------- privacy, comparison, export

def test_logs_contain_no_document_content_or_file_names(ctx, png_bytes):
    _run(ctx, [("patient_JohnDoe_scan.png", png_bytes)], ["mock_alpha"])
    for handler in logging.getLogger("rxbench").handlers:
        handler.flush()
    log_text = "".join(p.read_text(encoding="utf-8") for p in (ctx.manager.root.parent / "logs").glob("*.log"))
    assert "EXP_0001" in log_text                       # the run was logged ...
    for secret in ("JohnDoe", "patient_", "Mockamol", "Testex", "MASTER PROMPT", "Review after"):
        assert secret not in log_text                   # ... without names, output or prompt text


def test_comparison_highlights_differences_without_merging(ctx, png_bytes):
    _run(ctx, [("a.png", png_bytes)], ["mock_alpha", "mock_beta"])
    exp = ctx.manager.get("EXP_0001")
    report = write_comparison(ctx.manager, exp)
    assert report["caveat"] == AGREEMENT_CAVEAT and report["contains_mock_runs"]
    assert "medicine_strengths" in report["disagreements"]             # 650 vs 500 mg
    assert report["sections"]["follow_up_referral"]["status"] == "identical"
    assert set(report["sections"]["medicine_names"]["values"]) == {"mock_alpha", "mock_beta"}
    assert not any(k in report for k in ("consensus", "merged", "best_model"))
    md = (exp.path / "comparison" / "comparison_report.md").read_text(encoding="utf-8")
    assert "NOT evidence of correctness" in md and "MOCK" in md


def test_export_zip_and_metadata(ctx, png_bytes):
    _run(ctx, [("a.png", png_bytes)], ["mock_alpha"])
    exp = ctx.manager.get("EXP_0001")
    names = zipfile.ZipFile(io.BytesIO(export.experiment_zip_bytes(exp))).namelist()
    for expected in ("EXP_0001/input/original_prescription.png", "EXP_0001/prompt/master_prompt_snapshot.txt",
                     "EXP_0001/mock_alpha/raw_output.txt", "EXP_0001/mock_alpha/metadata.json",
                     "EXP_0001/comparison/comparison_report.json"):
        assert expected in names
    rows = export.collect_metadata(ctx.manager)
    assert export.metadata_csv(rows).splitlines()[0].startswith("experiment_id,batch_id,model_key")
    report = export.runtime_report_markdown(rows)
    assert "MOCK" in report and "mock_alpha" in report
