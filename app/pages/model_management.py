"""Screen 2: model management."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.config import model_config  # noqa: E402
from rxbench.models import backend_manager  # noqa: E402

LOCAL_TEXT = {"complete": "Downloaded", "config_only": "Processor files only (no weights)",
              "partial": "Partly downloaded", "not_downloaded": "Not downloaded",
              "revision_mismatch": "A different revision is present"}


def _download_with_progress(ctx, spec, total_bytes: int) -> None:
    result: dict = {}

    def work() -> None:
        try:
            result["value"] = ctx.downloads.download(spec, confirmed=True)
        except Exception as exc:
            result["error"] = str(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    bar = st.progress(0.0, text="Starting download…")
    while thread.is_alive():
        done = ctx.downloads.downloaded_bytes(spec)
        bar.progress(min(1.0, done / total_bytes) if total_bytes else 0.0,
                     text=f"{done / 1e9:.1f} / {total_bytes / 1e9:.1f} GB")
        time.sleep(2)
    if "error" in result:
        st.error(result["error"])
    else:
        bar.progress(1.0, text="Download finished")
        st.success(f"Local status: {result['value'].get('status')}, revision {result['value'].get('revision')}")


def _model_card(ctx, spec) -> None:
    local = ctx.downloads.local_status(spec)
    state = local.get("state") or {}
    precision = model_config(ctx.cfg, spec.key).get("precision") or spec.default_precision
    estimate = spec.memory_estimate_gb(precision)
    compat = backend_manager.check_model(spec)
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.subheader(spec.display_name)
            st.write(f"[{spec.repo_id}]({spec.repo_url})  ·  revision `{(spec.revision or 'main')[:12]}`")
            params = f"{spec.parameters_total / 1e9:.1f}B parameters"
            if spec.is_moe:
                params += (f" in total; about {spec.parameters_active / 1e9:.0f}B active per token "
                           "(mixture of experts: memory is sized on the total)")
            st.write(f"{params}  ·  image input: {'yes' if spec.image_input else 'no'}  ·  tier: {spec.tier}")
            st.write(f"Licence: [{spec.licence}]({spec.licence_url})  ·  access: {spec.access_requirements}")
            if spec.notes:
                st.caption(spec.notes)
        with right:
            st.metric("Local availability", LOCAL_TEXT.get(local["status"], local["status"]))
            st.write(f"Download size: {spec.download_size_gb} GB")
            st.write(f"Memory estimate at {precision}: about {estimate['total_gb']:.0f} GB "
                     f"({estimate['weights_gb']:.0f} GB weights + {estimate['runtime_overhead_gb']:.0f} GB runtime)")
            st.write("Backend compatibility: " + ("compatible" if compat["compatible"] else "NOT compatible"))
            load = state.get("last_load")
            st.write("Load status: " + ("never loaded on this machine" if not load else
                                        f"{'loaded successfully' if load['ok'] else 'load FAILED'} at {load['at']}"))
            ver = state.get("verification")
            st.write("File verification: " + ("not run" if not ver else
                                              f"{'passed' if ver['ok'] else 'FAILED'} at {ver['checked_at']}"))

        with st.expander("Details: precision options, backend checks, manual download"):
            st.dataframe([{"precision": p, **{k: v for k, v in spec.memory_estimate_gb(p).items() if k != "basis"},
                           "kind": "quantised on load" if p in spec.quantisation_options else "floating point"}
                          for p in spec.supported_precisions()], hide_index=True)
            st.caption("Estimates: total parameters × bytes per parameter + a runtime allowance. Measured values "
                       "are stored with every experiment.")
            st.dataframe(compat["checks"], hide_index=True)
            st.code(ctx.downloads.manual_download_command(spec), language="bash")
            if local.get("path"):
                st.write(f"Local path: `{local['path']}`")

        c1, c2, c3 = st.columns(3)
        if c1.button("Check access", key=f"access_{spec.key}"):
            access = ctx.downloads.access_status(spec)
            (st.success if access.get("ok") else st.warning)(access["detail"])
        if c2.button("Verify local files", key=f"verify_{spec.key}", disabled=local["status"] == "not_downloaded"):
            with st.spinner("Verifying…"):
                result = ctx.downloads.verify(spec)
            (st.success if result["ok"] else st.error)("Verification " + ("passed" if result["ok"] else "failed"))
            st.dataframe(result["checks"], hide_index=True)
        if local["status"] != "complete":
            if c3.button("Prepare download", key=f"plan_{spec.key}"):
                st.session_state[f"plan_{spec.key}_data"] = ctx.downloads.plan(spec)
            plan = st.session_state.get(f"plan_{spec.key}_data")
            if plan:
                st.write(f"**{plan['display_name']}** from {plan['repo_url']} at revision `{plan['revision']}`: "
                         f"{plan['download_bytes'] / 1e9:.1f} GB ({plan['size_source']}), "
                         f"{plan['already_present_bytes'] / 1e9:.1f} GB already present. "
                         f"Destination `{plan['cache_dir']}` has {plan['free_bytes'] / 1e9:.1f} GB free.")
                if plan.get("remote_error"):
                    st.warning(f"The Hub could not be reached: {plan['remote_error']}")
                if not plan["enough_space"]:
                    st.error("Not enough free disk space for this download.")
                confirmed = st.checkbox(f"I confirm downloading {plan['remaining_bytes'] / 1e9:.1f} GB and accept "
                                        f"the licence terms ({spec.licence}).", key=f"confirm_{spec.key}")
                if st.button("Download now", key=f"download_{spec.key}", type="primary",
                             disabled=not (confirmed and plan["enough_space"])):
                    _download_with_progress(ctx, spec, int(plan["download_bytes"]))
                    st.session_state.pop(f"plan_{spec.key}_data", None)


def render() -> None:
    ctx = common.get_ctx()
    common.header("Model management",
                  "Official checkpoints only, pinned to an exact revision. Nothing is downloaded without confirmation.")
    env_col, _ = st.columns([2, 1])
    with env_col:
        from rxbench.models.download_manager import hf_environment
        env = hf_environment()
        st.caption(f"Model cache: `{ctx.downloads.effective_cache_dir()}`  ·  Hugging Face token: "
                   f"{'present' if env['token_present'] else 'not found (needed for gated MedGemma)'}  ·  "
                   "On a shared cluster, download with the Slurm job script instead of this page.")
    for tier, title in (("primary", "Primary models"), ("optional", "Optional models")):
        st.header(title)
        for key in ctx.registry.keys(tier=tier):
            _model_card(ctx, ctx.registry.get(key))


render()
