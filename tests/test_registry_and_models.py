import json

import pytest

from rxbench.config import PROJECT_ROOT, model_config
from rxbench.inference.generation import GenerationConfig
from rxbench.models import backend_manager
from rxbench.models.base import AdapterError
from rxbench.models.download_manager import ConfirmationRequired, DownloadError, DownloadManager
from rxbench.models.registry import ModelRegistry, load_adapter_class

PRIMARY = {"medgemma_27b": "google/medgemma-27b-it", "gemma4_31b": "google/gemma-4-31B-it",
           "qwen3vl_32b": "Qwen/Qwen3-VL-32B-Instruct"}
OPTIONAL = {"gemma4_26b_a4b": "google/gemma-4-26B-A4B-it", "qwen3_5_27b": "Qwen/Qwen3.5-27B",
            "medgemma_1_5_4b": "google/medgemma-1.5-4b-it"}


@pytest.fixture
def registry():
    return ModelRegistry.from_yaml(PROJECT_ROOT / "model_registry.yaml")


def test_exact_repositories(registry):
    assert {k: registry.get(k).repo_id for k in registry.keys(tier="primary")} == PRIMARY
    assert {k: registry.get(k).repo_id for k in registry.keys(tier="optional")} == OPTIONAL
    for spec in registry.all():
        assert spec.repo_url == f"https://huggingface.co/{spec.repo_id}"
        assert spec.image_input and spec.backend == "transformers"
        assert spec.revision and len(spec.revision) == 40           # pinned commit
        assert spec.licence and spec.access_requirements


def test_text_only_and_lookalike_checkpoints_are_forbidden(registry):
    assert "google/medgemma-27b-text-it" in registry.get("medgemma_27b").forbidden_repo_ids
    assert "Qwen/Qwen3-VL-32B-Thinking" in registry.get("qwen3vl_32b").forbidden_repo_ids
    entry = registry.get("medgemma_27b").to_dict()
    entry.pop("key")
    entry["repo_id"] = "google/medgemma-27b-text-it"
    with pytest.raises(ValueError):
        ModelRegistry().register("bad", entry)


def test_adapter_classes_resolve(registry):
    for spec in registry.all():
        assert load_adapter_class(spec.adapter).__name__.endswith("Adapter")


def test_moe_memory_uses_total_parameters(registry):
    moe = registry.get("gemma4_26b_a4b")
    assert moe.is_moe and moe.parameters_active < moe.parameters_total
    assert moe.weights_gb("bf16") == pytest.approx(moe.parameters_total * 2 / 1024**3)
    assert moe.weights_gb("bf16") > 40          # ~48 GB, not the ~7 GB the active count would suggest


def test_memory_estimates_scale_with_precision(registry):
    spec = registry.get("qwen3vl_32b")
    est = {p: spec.memory_estimate_gb(p)["total_gb"] for p in ("fp32", "bf16", "int8", "int4")}
    assert est["fp32"] > est["bf16"] > est["int8"] > est["int4"]


def test_new_model_can_be_registered_without_code_changes(registry):
    entry = registry.get("qwen3vl_32b").to_dict()
    entry.pop("key")
    entry.update(repo_id="example/future-vlm", repo_url="https://huggingface.co/example/future-vlm",
                 forbidden_repo_ids=[])
    spec = registry.register("future_vlm", entry)
    assert "future_vlm" in registry and spec.display_name


def test_registry_rejects_bad_entries(registry):
    base = registry.get("gemma4_31b").to_dict()
    base.pop("key")
    with pytest.raises(ValueError):
        ModelRegistry().register("x", {**base, "image_input": False})
    with pytest.raises(ValueError):
        ModelRegistry().register("x", {**base, "precision_options": ["bf16", "fp7"]})
    with pytest.raises(ValueError):
        ModelRegistry().register("x", {k: v for k, v in base.items() if k != "licence"})


def test_backend_knows_every_registered_architecture(registry):
    for spec in registry.all():
        result = backend_manager.check_model(spec)
        assert result["compatible"], result


def test_non_native_image_options_are_rejected(ctx):
    spec = ctx.registry.get("qwen3vl_32b")
    cls = load_adapter_class(spec.adapter)
    with pytest.raises(AdapterError):
        cls(spec=spec, model_cfg={"image_processing": {"crop_box": [0, 0, 1, 1]}}, download_manager=ctx.downloads)
    with pytest.raises(AdapterError):
        load_adapter_class(ctx.registry.get("gemma4_31b").adapter)(
            spec=ctx.registry.get("gemma4_31b"), model_cfg={"image_processing": {"max_soft_tokens": 999}},
            download_manager=ctx.downloads)


def test_default_config_sets_documented_image_options(ctx):
    assert model_config(ctx.cfg, "gemma4_31b")["image_processing"]["max_soft_tokens"] == 1120
    assert model_config(ctx.cfg, "medgemma_27b")["image_processing"]["do_pan_and_scan"] is False


# ---------------------------------------------------------------- generation settings

def test_default_generation_is_greedy():
    kw = GenerationConfig().to_transformers_kwargs()
    assert kw["do_sample"] is False
    assert kw["temperature"] is None and kw["top_p"] is None and kw["top_k"] is None   # checkpoint sampling defaults cannot leak in
    assert kw["max_new_tokens"] == 8192 and kw["repetition_penalty"] == 1.0


def test_sampling_and_stop_settings():
    kw = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9, stop_sequences=["</end>"]).to_transformers_kwargs()
    assert kw["do_sample"] and kw["temperature"] == 0.7 and kw["top_p"] == 0.9 and kw["stop_strings"] == ["</end>"]


def test_invalid_generation_settings():
    with pytest.raises(ValueError):
        GenerationConfig.from_dict({"do_sample": True, "temperature": 0})
    with pytest.raises(ValueError):
        GenerationConfig.from_dict({"max_new_tokens": 0})
    with pytest.raises(ValueError):
        GenerationConfig.from_dict({"beam_width": 4})


# ---------------------------------------------------------------- download manager (no network)

def _fake_snapshot(tmp_path, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
                   present=(0, 1), config=True, tokenizer=True):
    snap = tmp_path / "snap"
    snap.mkdir()
    if config:
        (snap / "config.json").write_text("{}")
    if tokenizer:
        (snap / "tokenizer.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}))
    for i in present:
        (snap / shards[i]).write_bytes(b"x" * 10)
    return snap


def test_snapshot_classification(tmp_path):
    assert DownloadManager.inspect_snapshot(_fake_snapshot(tmp_path))["status"] == "complete"


def test_snapshot_partial(tmp_path):
    info = DownloadManager.inspect_snapshot(_fake_snapshot(tmp_path, present=(0,)))
    assert info["status"] == "partial" and info["weight_files_missing"] == ["model-00002-of-00002.safetensors"]


def test_snapshot_config_only(tmp_path):
    assert DownloadManager.inspect_snapshot(_fake_snapshot(tmp_path, present=()))["status"] == "config_only"


def test_download_requires_confirmation_and_space(ctx, monkeypatch):
    dm, spec = ctx.downloads, ctx.registry.get("gemma4_31b")
    plan = {"enough_space": True, "cache_dir": "x", "remaining_bytes": 1, "free_bytes": 2}
    monkeypatch.setattr(dm, "local_status", lambda s: {"status": "not_downloaded"})
    monkeypatch.setattr(dm, "plan", lambda s, remote=None: plan)
    with pytest.raises(ConfirmationRequired) as err:
        dm.download(spec)                              # no confirmed=True
    assert err.value.plan is plan
    monkeypatch.setattr(dm, "plan", lambda s, remote=None: {**plan, "enough_space": False})
    with pytest.raises(DownloadError):
        dm.download(spec, confirmed=True)


def test_already_downloaded_model_is_not_downloaded_again(ctx, monkeypatch):
    dm, spec = ctx.downloads, ctx.registry.get("gemma4_31b")
    monkeypatch.setattr(dm, "local_status", lambda s: {"status": "complete", "revision": spec.revision})
    monkeypatch.setattr(dm, "plan", lambda *a, **k: pytest.fail("must not plan a download"))
    assert dm.download(spec, confirmed=True)["downloaded"] is False


def test_manual_download_command_pins_revision(ctx):
    spec = ctx.registry.get("medgemma_27b")
    assert ctx.downloads.manual_download_command(spec) == f"hf download google/medgemma-27b-it --revision {spec.revision}"
