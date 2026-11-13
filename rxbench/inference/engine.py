"""Sequential zero-shot inference engine.

For each selected model, in turn:

    record memory -> load -> for every experiment: run, save raw output,
    parse, save metadata -> unload -> release -> verify memory

Models never see each other's output: each run reads only the experiment's
stored page and prompt snapshot. One model failing (to load or on one page)
is recorded and the engine moves on.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable

from ..config import model_config
from ..hardware import memory as mem
from ..logging_utils import get_logger, short_hash
from ..models.base import ModelNotAvailableError
from ..models.registry import ModelRegistry
from ..storage import metadata as md
from ..storage.experiment_manager import ExperimentManager, utc_now
from .generation import GenerationConfig
from .output_parser import parse_output
from .runtime_metadata import environment_record

log = get_logger()
Progress = Callable[[str, dict[str, Any]], None]


class InferenceEngine:
    def __init__(self, cfg: dict[str, Any], registry: ModelRegistry, manager: ExperimentManager,
                 download_manager, hardware_report: dict[str, Any] | None = None):
        self.cfg = cfg
        self.registry = registry
        self.manager = manager
        self.download_manager = download_manager
        self.environment = environment_record(hardware_report)
        self.unified_memory = bool(self.environment.get("unified_memory"))
        if cfg.get("execution", {}).get("allow_simultaneous"):
            raise NotImplementedError(
                "Simultaneous model serving is not implemented in this baseline. Set "
                "execution.allow_simultaneous: false; models run one after another.")

    # ------------------------------------------------------------------ helpers
    def _exec_cfg(self) -> dict[str, Any]:
        return {**self.cfg.get("execution", {}), "unified_memory": self.unified_memory}

    def _already_complete(self, exp, model_key: str, gen: GenerationConfig, precision: str, revision: str | None) -> bool:
        run = self.manager.load_model_run(exp, model_key)
        meta = run["metadata"]
        if not meta or meta.get("completion_status") != md.STATUS_COMPLETE or run["raw_text"] is None:
            return False
        return (meta["prompt"]["sha256"] == exp.record["prompt"]["sha256"]
                and meta["generation"]["requested"] == gen.to_dict()
                and (meta.get("requested_precision") or precision) == precision
                and (revision is None or meta["model"]["revision"] == revision))

    def _save_failure(self, exp, spec, status: str, error: str, gen: GenerationConfig, batch_id: str | None,
                      started: str, runtime: dict[str, Any] | None = None, memory: dict[str, Any] | None = None):
        meta = md.build_metadata(experiment=exp.record, model=spec.to_dict(), status=status,
                                 generation_config=gen.to_dict(), hardware=self.environment, batch_id=batch_id,
                                 started_at=started, finished_at=utc_now(), runtime=runtime, memory=memory,
                                 errors=[error], is_mock=spec.is_mock)
        self.manager.save_model_run(exp, spec.key, metadata=meta)

    # ------------------------------------------------------------------ main loop
    def run(self, exp_ids: list[str], model_keys: list[str], gen: GenerationConfig,
            batch_id: str | None = None, progress: Progress | None = None,
            skip_completed: bool = False) -> dict[str, Any]:
        gen.validate()
        emit: Progress = progress or (lambda event, info: None)
        experiments = [self.manager.get(e) for e in exp_ids]
        specs = [self.registry.get(k) for k in model_keys]          # fail early on unknown keys
        batch_id = batch_id or self.manager.new_batch_id()
        batch: dict[str, Any] = {
            "batch_id": batch_id, "started_at": utc_now(), "finished_at": None, "execution": "sequential",
            "experiments": exp_ids, "models": model_keys, "generation": gen.to_dict(),
            "environment": self.environment, "model_records": [],
            "contains_mock_runs": any(s.is_mock for s in specs),
        }
        self.manager.save_batch(batch_id, batch)
        tolerance = int(float(self.cfg.get("execution", {}).get("unload_tolerance_gb", 4.0)) * 1024**3)
        interval = float(self.cfg.get("execution", {}).get("memory_sample_interval_s", 0.25))

        for spec in specs:
            rec: dict[str, Any] = {"model_key": spec.key, "is_mock": spec.is_mock, "started_at": utc_now(),
                                   "load_ok": False, "runs": {}, "errors": []}
            batch["model_records"].append(rec)
            emit("model_start", {"model": spec.key})
            mcfg = model_config(self.cfg, spec.key)
            precision = mcfg.get("precision") or spec.default_precision
            todo = experiments
            if skip_completed:
                todo = [e for e in experiments
                        if not self._already_complete(e, spec.key, gen, precision, spec.revision)]
                for e in experiments:
                    if e not in todo:
                        rec["runs"][e.exp_id] = "skipped_already_complete"
                if not todo:
                    emit("model_skipped", {"model": spec.key})
                    rec["finished_at"] = utc_now()
                    self.manager.save_batch(batch_id, batch)
                    continue

            before_load = mem.snapshot("before_load")
            monitor = mem.MemoryMonitor(interval, self.unified_memory).start()
            adapter = None
            started = utc_now()
            try:
                adapter = self.registry.create_adapter(spec.key, mcfg, self.download_manager, self._exec_cfg())
                emit("model_loading", {"model": spec.key})
                adapter.load_model()
                load_memory = monitor.summary()
                rec["load_ok"] = True
                rec["load_seconds"] = adapter.load_seconds
                if self.download_manager is not None and not spec.is_mock:
                    self.download_manager.record_load(spec, True, {"precision": precision,
                                                                   "load_seconds": adapter.load_seconds})
            except Exception as exc:   # one model failing must not stop the others
                monitor.stop()
                status = (md.STATUS_NOT_AVAILABLE if isinstance(exc, ModelNotAvailableError)
                          else md.STATUS_FAILED_LOAD)
                err = f"{type(exc).__name__}: {exc}"
                log.error("model load failed key=%s error=%s", spec.key, type(exc).__name__)
                rec["errors"].append(err)
                for exp in todo:
                    self._save_failure(exp, spec, status, err, gen, batch_id, started)
                    rec["runs"][exp.exp_id] = status
                if self.download_manager is not None and not spec.is_mock:
                    self.download_manager.record_load(spec, False, {"error": type(exc).__name__})
                if adapter is not None:
                    try:
                        adapter.unload_model()
                    except Exception:
                        pass
                mem.release_torch_memory()
                rec["finished_at"] = utc_now()
                self.manager.save_batch(batch_id, batch)
                emit("model_failed", {"model": spec.key, "error": err})
                continue

            runtime = adapter.collect_runtime_metadata()
            base_warnings: list[str] = []
            if runtime.get("offloaded_to_cpu_or_disk"):
                base_warnings.append("Part of the model was placed on CPU or disk by device_map; timings are "
                                     "not comparable with a fully GPU-resident run.")
            for n, exp in enumerate(todo, start=1):
                emit("inference_start", {"model": spec.key, "experiment": exp.exp_id, "index": n, "total": len(todo)})
                run_started = utc_now()
                monitor.window()
                try:
                    prepared = self.manager.load_prepared_image(exp)
                    prompt = self.manager.load_prompt(exp)
                    result = adapter.run(prepared, prompt, gen)
                    inference_memory = monitor.summary()
                    warnings = list(base_warnings)
                    if result.input_pixel_sha256 != exp.record["input"]["image"]["model_input_pixel_sha256"]:
                        warnings.append("PROTOCOL WARNING: the pixels given to the model differ from the "
                                        "experiment's recorded page.")
                    if result.prompt_sha256 != prompt.sha256:
                        warnings.append("PROTOCOL WARNING: the prompt given to the model differs from the "
                                        "experiment's prompt snapshot.")
                    structured = parse_output(result.answer_text if result.answer_text is not None
                                              else result.raw_text)
                    if structured.get("reasoning_block_present") and not result.thinking.get("enabled"):
                        warnings.append(
                            "The model produced a reasoning block although reasoning mode was not requested. "
                            "It is kept in the raw output and left out of the structured parse; it also "
                            "consumed part of the output-token limit.")
                    status = md.STATUS_TRUNCATED if result.truncated else md.STATUS_COMPLETE
                    memory = {"unified_memory": self.unified_memory, "before_load": before_load,
                              "load": load_memory, "inference": inference_memory,
                              "image_preprocessing": result.preprocessing_memory,
                              "after_unload": None, "after_unload_check": None}
                    meta = md.build_metadata(
                        experiment=exp.record, model=spec.to_dict(), status=status,
                        generation_config=gen.to_dict(), hardware=self.environment, batch_id=batch_id,
                        started_at=run_started, finished_at=utc_now(), result=result.to_dict(),
                        runtime=runtime, memory=memory, parse_status=structured["parse_status"],
                        warnings=warnings, is_mock=spec.is_mock)
                    meta["requested_precision"] = precision
                    self.manager.save_model_run(exp, spec.key, metadata=meta, raw_text=result.raw_text,
                                                raw_text_with_special_tokens=result.raw_text_with_special_tokens,
                                                structured=structured)
                    rec["runs"][exp.exp_id] = status
                    log.info("run saved exp=%s model=%s image=%s status=%s out_tokens=%s seconds=%.1f",
                             exp.exp_id, spec.key, short_hash(exp.record["input"]["original_sha256"]), status,
                             result.n_output_tokens, result.generation_seconds or 0.0)
                    emit("inference_done", {"model": spec.key, "experiment": exp.exp_id, "status": status,
                                            "seconds": result.generation_seconds})
                except Exception as exc:   # one page failing must not stop the rest
                    err = f"{type(exc).__name__}: {exc}"
                    log.error("inference failed exp=%s model=%s error=%s", exp.exp_id, spec.key, type(exc).__name__)
                    log.debug("traceback: %s", traceback.format_exc())
                    self._save_failure(exp, spec, md.STATUS_FAILED_INFERENCE, err, gen, batch_id, run_started,
                                       runtime=runtime,
                                       memory={"unified_memory": self.unified_memory, "before_load": before_load,
                                               "load": load_memory, "inference": monitor.summary()})
                    rec["runs"][exp.exp_id] = md.STATUS_FAILED_INFERENCE
                    rec["errors"].append(f"{exp.exp_id}: {err}")
                    mem.release_torch_memory()
                    emit("inference_failed", {"model": spec.key, "experiment": exp.exp_id, "error": err})
            monitor.stop()

            emit("model_unloading", {"model": spec.key})
            try:
                adapter.unload_model()
            finally:
                del adapter
                mem.release_torch_memory()
            after_unload = mem.snapshot("after_unload")
            check = mem.released_after_unload(before_load, after_unload, tolerance, self.unified_memory)
            rec.update(after_unload_check=check, finished_at=utc_now())
            patch_warning = ([] if check.get("verified") in (True, None) else
                             ["Memory did not return to the pre-load level after unloading "
                              f"(residual {check.get('residual_bytes', 0) / 1024**3:.1f} GB)."])
            for exp in todo:
                run = self.manager.load_model_run(exp, spec.key)
                if run["metadata"] and run["metadata"].get("memory") is not None:
                    memory = run["metadata"]["memory"]
                    memory.update(after_unload=after_unload, after_unload_check=check)
                    self.manager.update_metadata(exp, spec.key, {
                        "memory": memory, "warnings": run["metadata"].get("warnings", []) + patch_warning})
            self.manager.save_batch(batch_id, batch)
            emit("model_done", {"model": spec.key, "released": check.get("verified")})

        batch["finished_at"] = utc_now()
        self.manager.save_batch(batch_id, batch)
        emit("batch_done", {"batch_id": batch_id})
        return batch
