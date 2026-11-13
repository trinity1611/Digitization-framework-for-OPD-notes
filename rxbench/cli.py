"""Command line interface: python -m rxbench <command> ...

Heavy libraries (torch, transformers, huggingface_hub) are imported only by the
commands that need them, so `hardware-report` works before the ML environment exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, model_config


def _gb(n: float | None) -> str:
    return "n/a" if n is None else f"{n / 1e9:.1f} GB"


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _ctx(args, enable_mock: bool | None = None):
    from .context import build_context
    ctx = build_context(args.config, enable_mock=enable_mock)
    for item in getattr(args, "precision", None) or []:          # e.g. --precision medgemma_27b=int4
        key, _, value = item.partition("=")
        spec = ctx.registry.get(key)
        if value not in spec.supported_precisions():
            raise SystemExit(f"{key} does not support precision '{value}' (supported: {spec.supported_precisions()})")
        ctx.cfg["models"].setdefault(key, {})["precision"] = value
    return ctx


def _gen_from_args(ctx, args):
    from .inference.generation import GenerationConfig
    d = dict(ctx.cfg["generation"])
    for name in ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed"):
        if getattr(args, name, None) is not None:
            d[name] = getattr(args, name)
    if getattr(args, "sample", False):
        d["do_sample"] = True
    if getattr(args, "enable_thinking", False):
        d["enable_thinking"] = True
    if getattr(args, "stop", None):
        d["stop_sequences"] = list(args.stop)
    return GenerationConfig.from_dict(d)


def _progress(event: str, info: dict[str, Any]) -> None:
    shown = {k: v for k, v in info.items() if k != "error"}
    line = f"[{event}] " + " ".join(f"{k}={v}" for k, v in shown.items())
    if "error" in info:
        line += f" error={str(info['error'])[:200]}"
    print(line, flush=True)


def _storage_paths(ctx) -> dict[str, Any]:
    return {"project": PROJECT_ROOT, "experiments": ctx.manager.root, "model_cache": ctx.downloads.effective_cache_dir()}


# --------------------------------------------------------------------------- commands

def cmd_hardware_report(args) -> int:
    from .hardware.diagnostics import write_report
    paths: dict[str, Any] = {"project": PROJECT_ROOT}
    try:
        ctx = _ctx(args)
        paths = _storage_paths(ctx)
    except Exception as exc:        # the report must still work in a bare Python
        print(f"note: application context not available ({type(exc).__name__}); reporting basic paths only",
              file=sys.stderr)
    out = Path(args.out) if args.out else PROJECT_ROOT / "docs" / "HARDWARE_REPORT.md"
    report, md = write_report(out, paths, args.role_note)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(md)
    print(f"\nWritten to {out}", file=sys.stderr)
    return 0


def cmd_models(args) -> int:
    ctx = _ctx(args)
    dm, reg = ctx.downloads, ctx.registry
    if args.action == "list":
        from .models.backend_manager import check_model
        print(f"{'key':<18}{'tier':<10}{'params':>8}  {'size':>9}  {'bf16 est.':>9}  {'local':<16}{'backend ok':<11}repository")
        for spec in reg.all():
            st = dm.local_status(spec)["status"]
            ok = check_model(spec)["compatible"]
            print(f"{spec.key:<18}{spec.tier:<10}{spec.parameters_total / 1e9:>7.1f}B  "
                  f"{(spec.download_size_gb or 0):>6.1f} GB  {spec.memory_estimate_gb()['total_gb']:>6.0f} GB  "
                  f"{st:<16}{'yes' if ok else 'NO':<11}{spec.repo_url}")
        return 0
    keys = ctx.resolve_models(args.models)
    rc = 0
    for key in keys:
        spec = reg.get(key)
        if args.action == "info":
            from .models.backend_manager import check_model
            _print({"registry": spec.to_dict(), "memory_estimates": {p: spec.memory_estimate_gb(p) for p in
                                                                    spec.supported_precisions()},
                    "local": dm.local_status(spec), "backend": check_model(spec),
                    "manual_download": dm.manual_download_command(spec)})
        elif args.action == "access":
            _print({key: dm.access_status(spec)})
        elif args.action == "plan":
            _print(dm.plan(spec))
        elif args.action == "download":
            from .models.download_manager import ConfirmationRequired, DownloadError
            try:
                if args.config_only:
                    result = dm.download(spec, config_only=True, progress=print)
                else:
                    plan = dm.plan(spec)
                    print(f"\n{spec.display_name}\n  repository : {spec.repo_url}\n  revision   : {spec.revision}\n"
                          f"  licence    : {spec.licence} ({spec.licence_url})\n  access     : {spec.access_requirements}\n"
                          f"  download   : {_gb(plan['download_bytes'])} ({plan['size_source']}), "
                          f"{_gb(plan['already_present_bytes'])} already present\n"
                          f"  destination: {plan['cache_dir']} ({_gb(plan['free_bytes'])} free)\n"
                          f"  local state: {plan['local_status']}")
                    if plan["local_status"] == "complete":
                        print("  already downloaded; nothing to do.")
                        continue
                    confirmed = args.yes
                    if not confirmed:
                        if not sys.stdin.isatty():
                            print("  not confirmed: pass --yes to confirm a download in a non-interactive session.")
                            rc = 2
                            continue
                        confirmed = input("  Download now? [y/N] ").strip().lower() in ("y", "yes")
                    if not confirmed:
                        print("  skipped.")
                        continue
                    result = dm.download(spec, confirmed=True, max_workers=args.workers, progress=print)
                print(f"  result: {result.get('status')} revision={result.get('revision')}")
            except (ConfirmationRequired, DownloadError) as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                rc = 1
        elif args.action == "verify":
            result = dm.verify(spec, check_remote_sizes=args.check_remote, deep_hash=args.deep_hash)
            print(f"{key}: {'OK' if result['ok'] else 'FAILED'}")
            for c in result["checks"]:
                print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
            rc = rc or (0 if result["ok"] else 1)
    return rc


def cmd_preflight(args) -> int:
    """Processor-only check: no weights are loaded."""
    from .inference.image_loader import prepare_image_from_bytes
    from .inference.prompt_loader import load_prompt
    ctx = _ctx(args)
    prompt = load_prompt(ctx.prompt_path)
    prepared = prepare_image_from_bytes(Path(args.image).read_bytes(), ctx.cfg["image"]["apply_exif_orientation"])
    gen = _gen_from_args(ctx, args)
    out: dict[str, Any] = {"note": "Processor-level preflight only: no model weights were loaded and nothing was generated.",
                           "prompt_sha256": prompt.sha256, "image": prepared.describe(), "models": {}}
    rc = 0
    for key in ctx.resolve_models(args.models):
        try:
            adapter = ctx.registry.create_adapter(key, model_config(ctx.cfg, key), ctx.downloads, ctx.cfg["execution"])
            out["models"][key] = adapter.preflight(prepared, prompt, gen)
        except Exception as exc:
            out["models"][key] = {"error": f"{type(exc).__name__}: {exc}"}
            rc = 1
    _print(out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return rc


def _collect_inputs(args) -> list[Path]:
    from .inference.image_loader import SUPPORTED_EXTENSIONS
    files = [Path(p) for p in (args.input or [])]
    if args.input_dir:
        files += sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if args.limit:
        files = files[: args.limit]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit(f"Input file(s) not found: {missing}")
    if not files:
        raise SystemExit("No input files. Use --input FILE... or --input-dir DIR.")
    return files


def cmd_run(args) -> int:
    if not args.mock:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")       # inference never needs the Hub
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ctx = _ctx(args, enable_mock=True if args.mock else None)
    from .inference.protocol import run_protocol
    keys = ["mock_alpha", "mock_beta"] if args.mock and not args.models else ctx.resolve_models(args.models)
    if args.mock:
        print("MOCK MODE: outputs come from a software test double, not from any model.", file=sys.stderr)
    files = [(p.name, p.read_bytes()) for p in _collect_inputs(args)]
    print(f"Every selected model processes the same original image and the same prompt independently.\n"
          f"documents={len(files)} models={keys}", flush=True)
    batch = run_protocol(ctx, files, keys, _gen_from_args(ctx, args), pages=args.pages, progress=_progress)
    return _summarise_batch(batch)


def cmd_rerun(args) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ctx = _ctx(args)
    from .inference.protocol import run_models
    exp_ids = ctx.manager.list_ids() if args.experiments == ["all"] else args.experiments
    batch = run_models(ctx, exp_ids, ctx.resolve_models(args.models), _gen_from_args(ctx, args),
                       progress=_progress, skip_completed=args.skip_completed)
    return _summarise_batch(batch)


def _summarise_batch(batch: dict[str, Any]) -> int:
    print(f"\nBatch {batch['batch_id']}")
    failed = 0
    for rec in batch["model_records"]:
        counts: dict[str, int] = {}
        for status in rec["runs"].values():
            counts[status] = counts.get(status, 0) + 1
        failed += sum(v for k, v in counts.items() if k.startswith(("failed", "model_not")))
        print(f"  {rec['model_key']}{' (MOCK)' if rec['is_mock'] else ''}: load_ok={rec['load_ok']} "
              + " ".join(f"{k}={v}" for k, v in counts.items()))
        for err in rec["errors"][:3]:
            print(f"    error: {err[:300]}")
    return 1 if failed else 0


def cmd_list(args) -> int:
    ctx = _ctx(args)
    for exp in ctx.manager.list_experiments():
        models = ", ".join(f"{k}:{(ctx.manager.load_model_run(exp, k)['metadata'] or {}).get('completion_status')}"
                           for k in ctx.manager.model_keys(exp))
        print(f"{exp.exp_id}  {exp.record['created_at']}  sha256={exp.record['input']['original_sha256'][:12]}  {models}")
    return 0


def cmd_compare(args) -> int:
    ctx = _ctx(args)
    from .evaluation.comparison import render_markdown, write_comparison
    ids = ctx.manager.list_ids() if args.experiments == ["all"] else args.experiments
    for exp_id in ids:
        report = write_comparison(ctx.manager, ctx.manager.get(exp_id))
        if len(ids) == 1:
            print(render_markdown(report))
        else:
            print(f"{exp_id}: differing sections: {report['disagreements']}")
    return 0


def cmd_reparse(args) -> int:
    """Rebuild structured_output.json from the stored raw output (the raw output is not touched)."""
    ctx = _ctx(args)
    from .evaluation.comparison import write_comparison
    from .inference.output_parser import parse_output
    from .storage.experiment_manager import STRUCTURED_FILE, utc_now, write_json
    ids = ctx.manager.list_ids() if args.experiments == ["all"] else args.experiments
    for exp_id in ids:
        exp = ctx.manager.get(exp_id)
        for key in ctx.manager.model_keys(exp):
            run = ctx.manager.load_model_run(exp, key)
            if run["raw_text"] is None:
                continue
            structured = parse_output(run["raw_text"])
            write_json(run["dir"] / STRUCTURED_FILE, structured)
            output = dict((run["metadata"] or {}).get("output") or {})
            output.update(parse_status=structured["parse_status"], reparsed_at=utc_now(),
                          parser_version=structured["parser"]["version"])
            ctx.manager.update_metadata(exp, key, {"output": output})
            print(f"{exp_id} {key}: {structured['parse_status']['method']} "
                  f"(missing sections: {structured['parse_status']['missing_sections'] or 'none'})")
        write_comparison(ctx.manager, exp)
    return 0


def cmd_annotation_template(args) -> int:
    ctx = _ctx(args)
    from .evaluation.annotation import empty_annotation, empty_region
    exp = ctx.manager.get(args.experiment)
    ann = empty_annotation(exp.record["input"]["original_sha256"], exp.record["input"]["original_filename"])
    ann["regions"] = [empty_region(1)]
    out = Path(args.out) if args.out else ctx.annotations_dir / f"{ann['image_sha256']}.json"
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Blank annotation written to {out}. It contains no model output; fill it in from the image.")
    return 0


def cmd_evaluate(args) -> int:
    ctx = _ctx(args)
    from .evaluation.evaluate import AnnotationError, evaluate_experiment, render_markdown
    exp = ctx.manager.get(args.experiment)
    path = Path(args.annotation) if args.annotation else ctx.annotations_dir / f"{exp.record['input']['original_sha256']}.json"
    if not path.exists():
        print(f"No annotation found at {path}. Create one with: python -m rxbench annotation-template {exp.exp_id}",
              file=sys.stderr)
        return 1
    try:
        report = evaluate_experiment(ctx.manager, exp, json.loads(path.read_text(encoding="utf-8")),
                                     allow_unverified=args.allow_unverified)
    except AnnotationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(render_markdown(report))
    return 0


def cmd_report(args) -> int:
    ctx = _ctx(args)
    from .storage.export import collect_metadata, runtime_report_markdown
    rows = collect_metadata(ctx.manager)
    if args.batch:
        rows = [r for r in rows if r.get("batch_id") == args.batch]
    md = runtime_report_markdown(rows, "Memory and inference-time report" + (f" ({args.batch})" if args.batch else ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


def cmd_export(args) -> int:
    ctx = _ctx(args)
    from .storage import export
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.what == "experiment":
        out.write_bytes(export.experiment_zip_bytes(ctx.manager.get(args.experiment)))
    else:
        rows = export.collect_metadata(ctx.manager, [args.experiment] if args.experiment else None)
        out.write_text(export.metadata_csv(rows) if out.suffix.lower() == ".csv" else export.metadata_json(rows),
                       encoding="utf-8", newline="")
    print(f"Written {out}")
    return 0


def cmd_delete(args) -> int:
    ctx = _ctx(args)
    if not args.yes:
        print("Refusing to delete without --yes.", file=sys.stderr)
        return 2
    for exp_id in args.experiments:
        ctx.manager.delete(exp_id)
        print(f"deleted {exp_id}")
    return 0


def cmd_offline_check(args) -> int:
    ctx = _ctx(args, enable_mock=True if args.mock else None)
    from . import offline
    keys = ["mock_alpha"] if args.mock and not args.models else ctx.resolve_models(args.models)
    report = offline.run_offline_check(
        cfg=ctx.cfg, registry=ctx.registry, download_manager=ctx.downloads, manager=ctx.manager,
        prompt_path=ctx.prompt_path, model_keys=keys, image_path=Path(args.image) if args.image else None,
        inference=args.inference, probe=not args.no_probe)
    md = offline.render_markdown(report)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8", newline="\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0 if report["no_network_use_attempted"] else 1


# --------------------------------------------------------------------------- parser

def _add_generation_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("generation (defaults come from config.yaml: greedy decoding)")
    g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--sample", action="store_true", help="enable sampling (default: greedy)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--repetition-penalty", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--stop", action="append", help="stop sequence (repeatable)")
    g.add_argument("--enable-thinking", action="store_true", help="enable reasoning mode where the model has one")
    p.add_argument("--precision", action="append", metavar="MODEL=PRECISION",
                   help="override a model's precision for this run, e.g. medgemma_27b=int4 (repeatable; recorded)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rxbench", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml (default: project config.yaml or $RXBENCH_CONFIG)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hardware-report", help="inspect this machine and write docs/HARDWARE_REPORT.md")
    p.add_argument("--out")
    p.add_argument("--json", help="also write the full report as JSON to this path")
    p.add_argument("--role-note", help="a sentence placed at the top (for example which machine this is)")
    p.set_defaults(func=cmd_hardware_report)

    p = sub.add_parser("models", help="list, inspect, download and verify models")
    p.add_argument("action", choices=["list", "info", "access", "plan", "download", "verify"])
    p.add_argument("models", nargs="?", help="'primary', 'optional', 'all' or comma-separated keys")
    p.add_argument("--yes", action="store_true", help="confirm the download (required when not interactive)")
    p.add_argument("--config-only", action="store_true", help="download processor/tokenizer/config files only")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--check-remote", action="store_true", help="verify: compare file sizes with the Hub")
    p.add_argument("--deep-hash", action="store_true", help="verify: also compare SHA-256 with the Hub (slow)")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("preflight", help="processor-only check that an image and the prompt reach each model's input")
    p.add_argument("--image", required=True)
    p.add_argument("--models", default="primary")
    p.add_argument("--out")
    _add_generation_args(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("run", help="run the zero-shot protocol on documents")
    p.add_argument("--input", nargs="+", help="image or PDF file(s)")
    p.add_argument("--input-dir", help="directory of images/PDFs")
    p.add_argument("--limit", type=int, help="use only the first N files")
    p.add_argument("--pages", help="PDF pages: 'all' (default), '1', '1,3', '2-4'")
    p.add_argument("--models", help="'primary' (default), 'optional', 'all' or comma-separated keys")
    p.add_argument("--mock", action="store_true", help="software test with mock adapters (NOT model inference)")
    _add_generation_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("rerun", help="run models again on existing experiments (earlier outputs go to history/)")
    p.add_argument("experiments", nargs="+", help="experiment ids, or 'all'")
    p.add_argument("--models", default="primary")
    p.add_argument("--skip-completed", action="store_true", help="skip runs already complete with the same settings")
    _add_generation_args(p)
    p.set_defaults(func=cmd_rerun)

    p = sub.add_parser("list", help="list experiments")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("compare", help="write the comparison report for experiments")
    p.add_argument("experiments", nargs="+", help="experiment ids, or 'all'")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("reparse", help="rebuild structured_output.json from stored raw outputs (no inference)")
    p.add_argument("experiments", nargs="+", help="experiment ids, or 'all'")
    p.set_defaults(func=cmd_reparse)

    p = sub.add_parser("annotation-template", help="write a blank ground-truth annotation for an experiment's image")
    p.add_argument("experiment")
    p.add_argument("--out")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_annotation_template)

    p = sub.add_parser("evaluate", help="evaluate an experiment against a human-verified annotation")
    p.add_argument("experiment")
    p.add_argument("--annotation")
    p.add_argument("--allow-unverified", action="store_true", help="provisional run on an unverified annotation")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="memory and inference-time report from stored metadata")
    p.add_argument("--batch")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="export an experiment (zip) or metadata (csv/json)")
    p.add_argument("what", choices=["experiment", "metadata"])
    p.add_argument("--experiment")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("delete", help="delete experiments")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("offline-check", help="verify that the pipeline runs with network access blocked")
    p.add_argument("--models", help="default: primary")
    p.add_argument("--image", help="a document to run through the pipeline")
    p.add_argument("--inference", action="store_true", help="also run real inference under the network guard")
    p.add_argument("--mock", action="store_true", help="software test of the check itself with a mock adapter")
    p.add_argument("--no-probe", action="store_true", help="do not probe whether the machine has network access")
    p.add_argument("--precision", action="append", metavar="MODEL=PRECISION")
    p.add_argument("--out")
    p.set_defaults(func=cmd_offline_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if args.command == "export" and args.what == "experiment" and not args.experiment:
        raise SystemExit("export experiment needs --experiment EXP_ID")
    if args.command == "models" and args.action != "list" and not args.models:
        raise SystemExit(f"models {args.action} needs a model selection (for example: primary)")
    return int(args.func(args) or 0)
