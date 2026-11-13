"""Document loading: original bytes are preserved, hashed, and decoded without alteration.

No OCR, cropping, layout analysis, region detection or enhancement happens
here or anywhere else in the application. The only operations applied before a
model's own processor are those needed to hand it a decoded upright RGB page:

* decoding the stored file,
* rendering a PDF page to pixels,
* applying the EXIF orientation tag (configurable),
* converting the colour mode to RGB.

Each one is recorded in `transformations`, and it is identical for every model.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_kind(filename: str, data: bytes) -> str:
    """Return 'pdf' or 'image' from the content signature, checked against the extension."""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext or filename}'. Accepted: JPG, JPEG, PNG, PDF.")
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    raise ValueError("File content is not a JPEG, PNG or PDF (signature check failed).")


def pdf_page_count(data: bytes) -> int:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_pdf_page(data: bytes, page_index: int, dpi: int = 200) -> Image.Image:
    """Render one PDF page (0-based) to an RGB image at the given resolution."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    try:
        if not 0 <= page_index < len(pdf):
            raise IndexError(f"PDF has {len(pdf)} page(s); page index {page_index} is out of range")
        page = pdf[page_index]
        bitmap = page.render(scale=dpi / 72.0)
        image = bitmap.to_pil().convert("RGB")
        image.load()
        return image
    finally:
        pdf.close()


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class PreparedImage:
    """The decoded page handed, unchanged, to every model's own processor."""
    image: Image.Image
    original_size: tuple[int, int]          # (width, height) as stored in the file
    original_mode: str
    original_format: str | None
    transformations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def pixel_sha256(self) -> str:
        """Hash of the exact pixels given to the processors (size + RGB bytes)."""
        h = hashlib.sha256()
        h.update(f"{self.image.width}x{self.image.height}:{self.image.mode}:".encode())
        h.update(self.image.tobytes())
        return h.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {"original_width": self.original_size[0], "original_height": self.original_size[1],
                "original_mode": self.original_mode, "original_format": self.original_format,
                "model_input_width": self.image.width, "model_input_height": self.image.height,
                "model_input_mode": self.image.mode, "model_input_pixel_sha256": self.pixel_sha256,
                "transformations": self.transformations}


def prepare_image_from_bytes(data: bytes, apply_exif_orientation: bool = True) -> PreparedImage:
    """Decode stored image bytes into the RGB page that all models receive."""
    img = Image.open(io.BytesIO(data))
    img.load()
    original_size, original_mode, original_format = img.size, img.mode, img.format
    transformations: list[dict[str, Any]] = []
    if apply_exif_orientation:
        orientation = img.getexif().get(0x0112)
        if orientation not in (None, 1):
            img = ImageOps.exif_transpose(img)
            transformations.append({"operation": "exif_orientation",
                                    "detail": f"EXIF orientation tag {orientation} applied; "
                                              f"size {original_size} -> {img.size}"})
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            # Composite transparency onto white so transparent paper is not turned black.
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            detail = f"mode {img.mode} -> RGB (alpha composited on white)"
            img = background
        else:
            detail = f"mode {img.mode} -> RGB"
            img = img.convert("RGB")
        transformations.append({"operation": "colour_mode_conversion", "detail": detail})
    return PreparedImage(image=img, original_size=original_size, original_mode=original_mode,
                         original_format=original_format, transformations=transformations)


def parse_page_selection(selection: str | None, n_pages: int) -> list[int]:
    """'all', '1', '1,3', '2-4' (1-based, as shown to users) -> sorted 0-based indices."""
    if selection is None or str(selection).strip().lower() in {"", "all"}:
        return list(range(n_pages))
    pages: set[int] = set()
    for part in str(selection).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise ValueError(f"Invalid page range '{part}'")
            pages.update(range(lo, hi + 1))
        else:
            pages.add(int(part))
    bad = [p for p in pages if not 1 <= p <= n_pages]
    if bad:
        raise ValueError(f"Page(s) {sorted(bad)} out of range; the document has {n_pages} page(s)")
    return sorted(p - 1 for p in pages)
