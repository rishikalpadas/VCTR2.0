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


# ---------------------------------------------------------------------------
# Colour separations
# ---------------------------------------------------------------------------
#
# One page per ink, for screen printing and spot-colour work. Every page has
# the same size, the artwork at the same offset and registration marks at the
# same positions, so the plates line up when the films are stacked.
#
# Plates are knocked out by default. VTracer's ``stacked`` mode paints each
# colour as a full blob and lets later colours cover it, so a plate made of
# just its own shapes would lay ink underneath every colour painted over it.
# Instead each plate replays the whole artwork in paint order: its own ink
# draws in colour, and anything of another colour painted *after* it draws in
# paper white, cutting it back out - the same thing the eye sees in the
# composite, without any boolean path geometry.

_SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NS)

_SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "text"}
# Subtrees that are referenced rather than painted where they sit.
_NON_RENDERED_TAGS = {"defs", "clipPath", "mask", "marker", "pattern", "symbol"}

# Enough for any palette the quantizer produces; a file with more distinct
# colours than this was not made for spot printing and would run to hundreds
# of pages.
MAX_SEPARATION_PLATES = 32

_PAPER = "#ffffff"
# An ink this light would be invisible drawn in its own colour on a white
# page (white ink for dark garments), so its plate is drawn in black instead.
_NEAR_WHITE_LUMA = 235

_PAGE_MARGIN = 48.0  # pt around the artwork, room for the registration marks
_HEADER_HEIGHT = 44.0  # pt above the margin for the plate label
_MARK_OFFSET = 20.0  # pt from the artwork corner to the mark centre


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _style_props(element: ET.Element) -> dict[str, str]:
    props: dict[str, str] = {}
    for item in (element.get("style") or "").split(";"):
        key, sep, value = item.partition(":")
        if sep:
            props[key.strip()] = value.strip()
    return props


def _normalize_paint(value: str | None) -> str | None:
    """A paint value as lowercase ``#rrggbb``, or None for no paint."""
    if value is None:
        return None
    value = value.strip()
    if value in ("", "none", "transparent"):
        return None
    if value.startswith("url("):
        raise InvalidSvgError(
            "The SVG uses a gradient or pattern fill, which cannot be split "
            "into flat colour separations."
        )

    from reportlab.lib.colors import toColor

    try:
        return "#" + toColor(value).hexval()[2:].lower()
    except Exception as exc:
        raise InvalidSvgError(
            f"The SVG uses a colour that could not be read: {value[:40]!r}."
        ) from exc


def _painted_shapes(root: ET.Element):
    """Yield ``(parent, element, fill, stroke)`` for every shape, in paint order.

    Resolves inheritance from ``<g>`` ancestors and ``style`` attributes, so
    the result is the colour each shape actually renders in. The SVG default
    paint is a black fill and no stroke.
    """

    def walk(parent, inherited_fill, inherited_stroke):
        for element in list(parent):
            tag = _local(element.tag)
            if tag in _NON_RENDERED_TAGS:
                continue
            props = _style_props(element)
            fill = props.get("fill", element.get("fill", inherited_fill))
            stroke = props.get("stroke", element.get("stroke", inherited_stroke))
            if tag in _SHAPE_TAGS:
                yield parent, element, fill, stroke
            else:
                yield from walk(element, fill, stroke)

    yield from walk(root, "black", "none")


def _luma(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def separation_colours(svg: str) -> list[str]:
    """The distinct inks in an SVG, darkest first - the plate order."""
    _reject_unsafe(svg)
    root = ET.fromstring(svg)
    found: set[str] = set()
    for _, _, fill, stroke in _painted_shapes(root):
        for paint in (_normalize_paint(fill), _normalize_paint(stroke)):
            if paint:
                found.add(paint)
    return sorted(found, key=lambda c: (_luma(c), c))


def _plate_svg(svg: str, colour: str, ink: str, knockout: bool) -> bytes:
    """The artwork reduced to one ink, as standalone SVG bytes."""
    root = ET.fromstring(svg)
    started = False  # has this ink painted anything yet?

    for parent, element, fill, stroke in list(_painted_shapes(root)):
        fill, stroke = _normalize_paint(fill), _normalize_paint(stroke)
        is_mine = colour in (fill, stroke)

        def resolve(paint):
            if paint == colour:
                return ink
            # Another colour on top of this ink cuts it back to paper. Below
            # it there is nothing to cut, so it is simply dropped.
            if paint and knockout and started:
                return _PAPER
            return None

        new_fill, new_stroke = resolve(fill), resolve(stroke)
        started = started or is_mine

        if new_fill is None and new_stroke is None:
            parent.remove(element)
            continue

        # Explicit attributes beat anything inherited from a group; a style
        # attribute on the element itself would beat them, so it goes.
        props = _style_props(element)
        props.pop("fill", None)
        props.pop("stroke", None)
        if props:
            element.set("style", ";".join(f"{k}:{v}" for k, v in props.items()))
        else:
            element.attrib.pop("style", None)
        element.set("fill", new_fill or "none")
        element.set("stroke", new_stroke or "none")

    return ET.tostring(root, encoding="utf-8")


def _draw_registration_mark(canvas, x: float, y: float) -> None:
    canvas.circle(x, y, 7, stroke=1, fill=0)
    canvas.line(x - 12, y, x + 12, y)
    canvas.line(x, y - 12, x, y + 12)


def svg_to_separations_pdf(svg: str, knockout: bool = True) -> bytes:
    """Split an SVG into one vector PDF page per colour.

    Pages run darkest ink first. Each is labelled with its order ("1st",
    "2nd", ...), a swatch and the hex value, and carries registration marks.
    ``knockout=False`` gives overprint plates instead: each colour's shapes
    whole, including the parts other colours cover.
    """
    colours = separation_colours(svg)
    if not colours:
        raise InvalidSvgError("The SVG has no coloured shapes to separate.")
    if len(colours) > MAX_SEPARATION_PLATES:
        raise InvalidSvgError(
            f"The artwork has {len(colours)} colours; colour separation "
            f"supports up to {MAX_SEPARATION_PLATES}. Vectorize it with a "
            f"smaller palette first."
        )

    try:
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import HexColor, black
        from reportlab.pdfgen.canvas import Canvas
        from svglib.svglib import svg2rlg
    except ImportError as exc:
        raise PdfExportError(
            "PDF export is not available on this server.",
            detail=f"svglib/reportlab import failed: {exc}",
        ) from exc

    drawings = []
    for colour in colours:
        ink = "#000000" if _luma(colour) >= _NEAR_WHITE_LUMA else colour
        try:
            drawing = svg2rlg(io.BytesIO(_plate_svg(svg, colour, ink, knockout)))
        except Exception as exc:
            log.exception("svglib failed to parse the %s plate", colour)
            raise PdfExportError(
                "The vector artwork could not be converted to separations.",
                detail=f"{colour}: {type(exc).__name__}: {exc}",
            ) from exc
        if drawing is None:
            raise PdfExportError(
                "The vector artwork could not be converted to separations.",
                detail=f"{colour}: svg2rlg returned no drawing",
            )
        drawings.append((colour, ink, drawing))

    # Every plate comes from the same SVG root, so they share one size.
    width = max(d.width for _, _, d in drawings)
    height = max(d.height for _, _, d in drawings)
    page_w = width + 2 * _PAGE_MARGIN
    page_h = height + 2 * _PAGE_MARGIN + _HEADER_HEIGHT
    left, bottom = _PAGE_MARGIN, _PAGE_MARGIN
    top = bottom + height

    buffer = io.BytesIO()
    pdf = Canvas(buffer, pagesize=(page_w, page_h), pageCompression=1)
    pdf.setTitle("Colour separations")

    for number, (colour, ink, drawing) in enumerate(drawings, start=1):
        label_y = top + _MARK_OFFSET + 18
        pdf.setFillColor(black)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(left, label_y, _ordinal(number))

        pdf.setFillColor(HexColor(colour))
        pdf.setStrokeColor(HexColor("#666666"))
        pdf.setLineWidth(0.5)
        pdf.rect(left + 44, label_y - 2, 16, 16, stroke=1, fill=1)

        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 10)
        note = "  (white ink, shown in black)" if ink != colour else ""
        pdf.drawString(left + 68, label_y + 3, colour.upper() + note)
        pdf.drawRightString(left + width, label_y + 3, f"Plate {number} of {len(drawings)}")

        renderPDF.draw(drawing, pdf, left, bottom)

        pdf.setStrokeColor(black)
        pdf.setLineWidth(0.5)
        for x in (left - _MARK_OFFSET, left + width + _MARK_OFFSET):
            for y in (bottom - _MARK_OFFSET, top + _MARK_OFFSET):
                _draw_registration_mark(pdf, x, y)

        pdf.showPage()

    pdf.save()
    result = buffer.getvalue()
    curves = _assert_vector_pdf(result)

    log.info(
        "Separation export: %d plates (%s), %.1f KB PDF, %.0fx%.0f pt, "
        "%d curve ops, knockout=%s",
        len(drawings),
        ", ".join(colours),
        len(result) / 1024,
        width,
        height,
        curves,
        knockout,
    )
    return result
