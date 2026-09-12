"""Export a generated SVG to other vector formats.

Currently PDF, via svglib + reportlab. Both are pure Python wheels, so this
adds no native dependency on Windows - which rules out the obvious
alternative, CairoSVG, whose cairo DLLs are the usual reason a Windows setup
fails.

The output is real vector artwork: measured on a badge, the PDF content stream
carries 1376 Bezier (`c`) operators and contains no image XObject at all. The
same anti-raster guarantee the SVG side makes is asserted here too, in
``_assert_vector_pdf``.

Trust boundary
--------------
The SVG handed to this module comes back from the browser, so it is treated as
untrusted input even though this service generated it moments earlier. Three
things are rejected before any parser sees it:

  * a DOCTYPE or ENTITY declaration - lxml (svglib's parser) will expand
    internal entities, which is the billion-laughs denial of service;
  * any ``href`` / ``xlink:href`` - svglib resolves those, including over the
    network, which would turn this endpoint into an SSRF gadget;
  * anything oversized.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET

from errors import InvalidSvgError, VectorizationError
from logging_config import get_logger

log = get_logger(__name__)

# Generous next to a realistic output (50-150 KB) but far below anything that
# could exhaust memory in the renderer.
MAX_SVG_BYTES = 8 * 1024 * 1024

_DOCTYPE_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_HREF_RE = re.compile(r"\b(?:xlink:)?href\s*=", re.IGNORECASE)


class PdfExportError(VectorizationError):
    code = "PDF_EXPORT_FAILED"
    http_status = 500


def _reject_unsafe(svg: str) -> None:
    if not svg or not svg.strip():
        raise InvalidSvgError("No SVG content was supplied.")

    if len(svg.encode("utf-8")) > MAX_SVG_BYTES:
        raise InvalidSvgError(
            f"The SVG is larger than the "
            f"{MAX_SVG_BYTES // (1024 * 1024)} MB export limit."
        )

    if _DOCTYPE_RE.search(svg):
        raise InvalidSvgError(
            "The SVG contains a DOCTYPE or entity declaration, which is not "
            "accepted for export."
        )

    if _HREF_RE.search(svg):
        raise InvalidSvgError(
            "The SVG references external resources, which are not accepted "
            "for export."
        )

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise InvalidSvgError(
            "The SVG could not be parsed.", detail=f"XML parse error: {exc}"
        ) from exc

    if not root.tag.endswith("svg"):
        raise InvalidSvgError("That file is not an SVG document.")

    if any(element.tag.endswith("image") for element in root.iter()):
        raise InvalidSvgError("The SVG embeds a raster image; refusing to export.")


def _assert_vector_pdf(pdf: bytes) -> int:
    """Confirm the PDF is drawn geometry, not a pasted bitmap.

    Returns the number of Bezier operators found, for logging.

    Note the ProcSet boilerplate (``/ProcSet [ /PDF /Text /ImageB /ImageC
    /ImageI ]``) contains the word "Image" in every reportlab PDF ever
    written - matching on that instead of ``/Subtype /Image`` gives a false
    positive every single time.
    """
    if not pdf.startswith(b"%PDF-"):
        raise PdfExportError("PDF export produced a file that is not a PDF.")

    if re.search(rb"/Subtype\s*/Image", pdf):
        raise PdfExportError(
            "PDF export embedded a raster image instead of vector paths."
        )

    curves = _count_curve_operators(pdf)
    if curves == 0:
        log.warning("Exported PDF contains no curve operators")
    return curves


def _count_curve_operators(pdf: bytes) -> int:
    """Best-effort count of Bezier operators across the content streams.

    reportlab writes streams as ASCII85 + Flate, so both layers have to come
    off before the operators are visible. Failure here is not fatal - it only
    costs a log line.
    """
    import base64
    import zlib

    total = 0
    for raw in re.findall(rb"stream\r?\n(.*?)endstream", pdf, re.S):
        body = raw.strip()
        if body.endswith(b"~>"):
            body = body[:-2]
        for decode in (
            lambda b: zlib.decompress(base64.a85decode(b, adobe=False)),
            zlib.decompress,
            lambda b: b,
        ):
            try:
                data = decode(body)
            except Exception:
                continue
            total += len(re.findall(rb"(?<![A-Za-z0-9])c(?![A-Za-z0-9])", data))
            break
    return total


def svg_to_pdf(svg: str) -> bytes:
    """Convert an SVG document to a single-page vector PDF.

    Page size follows the SVG's own dimensions. svglib converts CSS pixels to
    PostScript points at 0.75 (96 dpi -> 72 dpi), so a 902px-wide artwork
    becomes a 676.5pt page - 9.4 inches, which is the physically correct size
    rather than an arbitrary A4 fit.
    """
    _reject_unsafe(svg)

    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
    except ImportError as exc:
        raise PdfExportError(
            "PDF export is not available on this server.",
            detail=f"svglib/reportlab import failed: {exc}",
        ) from exc

    try:
        drawing = svg2rlg(io.BytesIO(svg.encode("utf-8")))
    except Exception as exc:
        log.exception("svglib failed to parse the SVG")
        raise PdfExportError(
            "The vector artwork could not be converted to PDF.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if drawing is None:
        raise PdfExportError(
            "The vector artwork could not be converted to PDF.",
            detail="svg2rlg returned no drawing",
        )

    buffer = io.BytesIO()
    try:
        renderPDF.drawToFile(drawing, buffer)
    except Exception as exc:
        log.exception("reportlab failed to render the PDF")
        raise PdfExportError(
            "The vector artwork could not be written as a PDF.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    pdf = buffer.getvalue()
    curves = _assert_vector_pdf(pdf)

    log.info(
        "PDF export: %.1f KB SVG -> %.1f KB PDF, %.0fx%.0f pt, %d curve ops",
        len(svg.encode("utf-8")) / 1024,
        len(pdf) / 1024,
        getattr(drawing, "width", 0),
        getattr(drawing, "height", 0),
        curves,
    )
    return pdf
