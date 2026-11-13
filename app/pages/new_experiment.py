"""Screen 3: new experiment."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.inference import image_loader  # noqa: E402
from rxbench.inference.generation import GenerationConfig  # noqa: E402
from rxbench.inference.prompt_loader import load_prompt  # noqa: E402
from rxbench.inference.protocol import create_experiments, run_models  # noqa: E402


def _generation_form(ctx) -> GenerationConfig | None:
    d = ctx.cfg["generation"]
    st.subheader("4. Generation configuration")
    c1, c2, c3 = st.columns(3)
    max_new = c1.number_input("Maximum new tokens", 256, 65536, int(d["max_new_tokens"]), step=1024,
                              help="Large enough for the full structured response. A truncated response is "
                                   "marked incomplete and can be rerun with a higher limit.")
    sampling = c2.toggle("Sampling", value=bool(d["do_sample"]), help="Off = greedy decoding (the baseline default).")
    seed = c3.number_input("Random seed", 0, 2**31 - 1, int(d["seed"] or 0))
    c1, c2, c3 = st.columns(3)
    temperature = c1.number_input("Temperature", 0.0, 2.0, float(d["temperature"]) if sampling else 0.0, step=0.1,
                                  disabled=not sampling, help="0 with sampling off = deterministic greedy decoding.")
    repetition = c2.number_input("Repetition penalty", 0.5, 2.0, float(d["repetition_penalty"]), step=0.05)
    thinking = c3.toggle("Reasoning / thinking mode", value=bool(d["enable_thinking"]),
                         help="Only for models that have one (Gemma 4, Qwen3.5). Off for the baseline.")
    stops = st.text_input("Stop sequences (comma-separated, optional)", ", ".join(d["stop_sequences"]))
    try:
        return GenerationConfig.from_dict({
            "do_sample": sampling, "temperature": temperature if sampling else 0.0, "top_p": d["top_p"],
            "top_k": d["top_k"], "max_new_tokens": int(max_new), "repetition_penalty": float(repetition),
            "seed": int(seed), "stop_sequences": [s.strip() for s in stops.split(",") if s.strip()],
            "enable_thinking": thinking})
    except ValueError as exc:
        st.error(str(exc))
        return None


def render() -> None:
    ctx = common.get_ctx()
    common.header("New experiment")
    st.warning(common.INDEPENDENCE_NOTICE)

    st.subheader("1. Document")
    upload = st.file_uploader("Prescription image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    pages: list[int] | None = None
    data = None
    if upload is not None:
        data = upload.getvalue()
        try:
            kind = image_loader.detect_kind(upload.name, data)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.caption(f"SHA-256 `{image_loader.sha256_hex(data)}`  ·  {len(data) / 1024:.0f} KB  ·  the original bytes "
                   "are stored unchanged")
        if kind == "pdf":
            n = image_loader.pdf_page_count(data)
            all_pages = st.checkbox(f"Process all {n} pages", value=True)
            chosen = list(range(1, n + 1)) if all_pages else st.multiselect("Pages", list(range(1, n + 1)), default=[1])
            pages = [p - 1 for p in chosen]
            preview_page = st.selectbox("Preview page", chosen or [1])
            st.image(image_loader.render_pdf_page(data, preview_page - 1, ctx.cfg["image"]["pdf_render_dpi"]),
                     caption=f"Page {preview_page} of {n}, rendered at {ctx.cfg['image']['pdf_render_dpi']} dpi", width=520)
        else:
            prepared = image_loader.prepare_image_from_bytes(data, ctx.cfg["image"]["apply_exif_orientation"])
            st.image(Image.open(io.BytesIO(data)), width=520,
                     caption=f"{upload.name}  ·  {prepared.original_size[0]}×{prepared.original_size[1]} px")
            if prepared.transformations:
                st.caption("Applied before every model's processor: "
                           + "; ".join(t["detail"] for t in prepared.transformations))
            else:
                st.caption("No transformation is applied before the models' own processors.")

    st.subheader("2. Models")
    all_keys = common.ordered_model_keys(ctx, ctx.registry.keys())
    status = {k: (ctx.downloads.local_status(ctx.registry.get(k))["status"] if not ctx.registry.get(k).is_mock
                  else "complete") for k in all_keys}
    selected = st.multiselect(
        "Models to run (one after another)", all_keys, default=ctx.registry.keys(tier="primary"),
        format_func=lambda k: f"{common.model_name(ctx, k)}  [{'ready' if status[k] == 'complete' else 'not downloaded'}]")
    missing = [common.model_name(ctx, k) for k in selected if status[k] != "complete"]
    if missing:
        st.warning("Not downloaded: " + ", ".join(missing) + ". They will be recorded as not run; no other "
                   "checkpoint is substituted. Download them on the Model management page.")
    if any(ctx.registry.get(k).is_mock for k in selected):
        st.error("A MOCK entry is selected. Mock output is a software test double, not model output.")

    st.subheader("3. Master prompt")
    prompt = load_prompt(ctx.prompt_path)
    st.caption(f"`{ctx.prompt_path.name}`  ·  {prompt.n_chars:,} characters  ·  SHA-256 `{prompt.sha256}`  ·  sent "
               "unchanged and in full to every model")
    if prompt.has_printed_text_exclusion:
        st.success("The printed / pre-filled text exclusion rule is present in the prompt.")
    else:
        st.error("The printed / pre-filled text exclusion rule is missing from the prompt. Experiments are blocked.")
    with st.expander("Review the master prompt"):
        st.text(prompt.text)

    gen = _generation_form(ctx)

    st.subheader("5. Start")
    ready = bool(data and selected and gen and prompt.has_printed_text_exclusion and (pages is None or pages))
    if st.button("Start experiment", type="primary", disabled=not ready):
        batch_id = ctx.manager.new_batch_id()
        exps = create_experiments(ctx, data, upload.name, prompt, pages, batch_id)
        exp_ids = [e.exp_id for e in exps]
        with st.status(f"Running {len(selected)} model(s) on {len(exp_ids)} page(s), sequentially…", expanded=True) as box:
            def progress(event: str, info: dict) -> None:
                messages = {
                    "model_loading": "Loading {model}…", "inference_start": "{model}: page {index} of {total}",
                    "inference_done": "{model}: {experiment} finished ({status})",
                    "inference_failed": "{model}: {experiment} FAILED: {error}",
                    "model_failed": "{model} could not be run: {error}",
                    "model_done": "{model} unloaded; memory released: {released}"}
                if event in messages:
                    box.write(messages[event].format(**{k: info.get(k) for k in
                                                        ("model", "index", "total", "experiment", "status", "error", "released")}))
            batch = run_models(ctx, exp_ids, selected, gen, batch_id=batch_id, progress=progress,
                               hardware_report=common.hardware_report())
            box.update(label=f"Finished: {batch['batch_id']}", state="complete")
        st.session_state["selected_experiment"] = exp_ids[0]
        st.success(f"Created {', '.join(exp_ids)}. Open **Results** or **Comparison** to inspect the outputs.")


render()
