import hashlib
import io

import pytest
from PIL import Image

from rxbench.inference import image_loader as il
from rxbench.inference.prompt_loader import load_prompt, prompt_from_bytes
from tests.conftest import image_bytes, make_image


# ---------------------------------------------------------------- master prompt

def test_master_prompt_is_complete_and_unmodified(ctx):
    p = load_prompt(ctx.prompt_path)
    assert p.sha256 == hashlib.sha256(ctx.prompt_path.read_bytes()).hexdigest()
    assert p.text == ctx.prompt_path.read_bytes().decode("utf-8")           # no stripping or rewriting
    assert p.text.startswith("# MASTER PROMPT: ZERO-SHOT HANDWRITTEN MEDICAL PRESCRIPTION TRANSCRIPTION")
    assert p.text.rstrip().endswith("faithful transcription, traceability and transparent uncertainty.")
    for heading in ("## ROLE", "# 1. PRIMARY OBJECTIVE", "# 3. WHOLE-PAGE UNDERSTANDING",
                    "# 4. VERBATIM HANDWRITING TRANSCRIPTION", "# 5. SPATIAL AND LAYOUT PRESERVATION",
                    "# 6. STRUCTURED CLINICAL CLASSIFICATION", "## 6.11 Other / unclassified",
                    "# 7. CLASSIFICATION EVIDENCE AND UNCERTAINTY", "# 8. REQUIRED OUTPUT FORMAT",
                    "## E. Final validation", "# 9. STRICT CONSTRAINTS"):
        assert heading in p.text, heading
    assert len(p.text) > 15000          # the full prompt, not a summary


def test_printed_text_exclusion_rule_present(ctx):
    p = load_prompt(ctx.prompt_path)
    assert p.has_printed_text_exclusion
    assert "# 2. CRITICAL RULE: IGNORE PRE-EXISTING PRINTED OR PRE-FILLED TEXT" in p.text
    assert "Do not ignore handwriting merely because it appears inside a printed field" in p.text


def test_prompt_without_rule_is_detected():
    p = prompt_from_bytes(b"Transcribe the handwriting.")
    assert not p.has_printed_text_exclusion


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        prompt_from_bytes(b"  \n")


# ---------------------------------------------------------------- images

def test_plain_image_is_not_altered(png_bytes):
    prepared = il.prepare_image_from_bytes(png_bytes)
    assert prepared.original_size == prepared.size == (320, 240)      # no resizing, no cropping
    assert prepared.transformations == []
    assert prepared.image.tobytes() == Image.open(io.BytesIO(png_bytes)).convert("RGB").tobytes()


def test_pixel_hash_is_deterministic(png_bytes):
    assert il.prepare_image_from_bytes(png_bytes).pixel_sha256 == il.prepare_image_from_bytes(png_bytes).pixel_sha256


def test_mode_conversion_is_recorded():
    prepared = il.prepare_image_from_bytes(image_bytes("PNG", mode="L"))
    assert prepared.image.mode == "RGB"
    assert [t["operation"] for t in prepared.transformations] == ["colour_mode_conversion"]


def test_exif_orientation_is_recorded_and_optional():
    img = make_image(320, 240)
    exif = Image.Exif()
    exif[0x0112] = 6                                            # rotate 90 degrees
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    rotated = il.prepare_image_from_bytes(buf.getvalue(), apply_exif_orientation=True)
    assert rotated.size == (240, 320) and rotated.original_size == (320, 240)
    assert rotated.transformations[0]["operation"] == "exif_orientation"
    untouched = il.prepare_image_from_bytes(buf.getvalue(), apply_exif_orientation=False)
    assert untouched.size == (320, 240) and untouched.transformations == []


def test_kind_detection_and_rejection(png_bytes, jpeg_bytes, pdf_bytes):
    assert il.detect_kind("a.png", png_bytes) == "image"
    assert il.detect_kind("a.JPEG", jpeg_bytes) == "image"
    assert il.detect_kind("a.jpg", jpeg_bytes) == "image"
    assert il.detect_kind("a.pdf", pdf_bytes) == "pdf"
    with pytest.raises(ValueError):
        il.detect_kind("a.tiff", png_bytes)
    with pytest.raises(ValueError):
        il.detect_kind("a.png", b"not an image")


def test_pdf_pages_render_in_order(pdf_bytes):
    assert il.pdf_page_count(pdf_bytes) == 3
    page = il.render_pdf_page(pdf_bytes, 1, dpi=100)
    assert page.mode == "RGB" and page.width > 0
    with pytest.raises(IndexError):
        il.render_pdf_page(pdf_bytes, 3)


@pytest.mark.parametrize("spec,expected", [(None, [0, 1, 2]), ("all", [0, 1, 2]), ("2", [1]), ("1,3", [0, 2]),
                                           ("2-3", [1, 2]), ("3,1-2", [0, 1, 2])])
def test_page_selection(spec, expected):
    assert il.parse_page_selection(spec, 3) == expected


def test_page_selection_out_of_range():
    with pytest.raises(ValueError):
        il.parse_page_selection("4", 3)
