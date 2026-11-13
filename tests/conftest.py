"""Shared fixtures. All tests here are software tests; none of them is model inference
unless the test name says `real_` and the required local files are present."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_image(width: int = 320, height: int = 240, text: str = "T. Test 500mg", mode: str = "RGB") -> Image.Image:
    img = Image.new(mode, (width, height), "white" if mode != "L" else 255)
    ImageDraw.Draw(img).text((10, 10), text, fill="black" if mode != "L" else 0)
    return img


def image_bytes(fmt: str = "PNG", **kwargs) -> bytes:
    buf = io.BytesIO()
    make_image(**kwargs).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return image_bytes("PNG")


@pytest.fixture
def jpeg_bytes() -> bytes:
    return image_bytes("JPEG")


@pytest.fixture
def pdf_bytes() -> bytes:
    pages = [make_image(400, 300, f"page {n}") for n in (1, 2, 3)]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """Application context in a temporary directory with two mock models registered."""
    monkeypatch.setenv("RXBENCH_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("RXBENCH_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("RXBENCH_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RXBENCH_ANNOTATIONS_DIR", str(tmp_path / "annotations"))
    monkeypatch.delenv("RXBENCH_CONFIG", raising=False)
    from rxbench.context import build_context
    return build_context(enable_mock=True)


@pytest.fixture
def prompt(ctx):
    from rxbench.inference.prompt_loader import load_prompt
    return load_prompt(ctx.prompt_path)


@pytest.fixture
def gen():
    from rxbench.inference.generation import GenerationConfig
    return GenerationConfig()
