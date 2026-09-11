"""SVG cleanup, optimization and validation.

Four jobs, in order:

1. **Normalize the root element.** VTracer emits ``width``/``height`` but no
   ``viewBox``, which makes the file non-scalable in some consumers and loses
   the link back to the original image size. We set a ``viewBox`` in processed
   pixel space and ``width``/``height`` in *original* image units, so the SVG
   scales cleanly and reports the dimensions the user uploaded.

2. **Cull tracing artifacts.** Tiny stray paths - single anti-aliased pixels,
   JPEG noise - carry no visual information but inflate the file and clutter
   the layer list in Illustrator. Paths whose bounding-box diagonal is under a
   fraction of the canvas diagonal get dropped.

3. **Optimize with scour.** Numeric precision reduction, metadata and comment
   removal, attribute tidying. Deliberately *not* maximum aggression: group
   collapsing and id stripping are fine, but dropping below 5 significant
   digits snaps every control point to the pixel grid - see OptimizeParams.

4. **Validate.** The output must parse as XML, must contain real ``<path>``
   geometry, and must NOT contain an embedded raster. That last check is the
   one that makes "it has a .svg extension" not good enough.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from errors import InvalidSvgError
from logging_config import get_logger
from presets import OptimizeParams

log = get_logger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

_NUMBER_RE = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_TRANSLATE_RE = re.compile(
    r"translate\(\s*(-?[\d.eE+-]+)[\s,]+(-?[\d.eE+-]+)\s*\)"
)


@dataclass
class OptimizeResult:
    svg: str
    path_count: int
    removed_paths: int
    bytes_before: int
    bytes_after: int
    warnings: list[str]


def finalize_svg(
    raw_svg: str,
    *,
    processed_width: int,
    processed_height: int,
    original_width: int,
    original_height: int,
    params: OptimizeParams,
) -> OptimizeResult:
    bytes_before = len(raw_svg.encode("utf-8"))
    warnings: list[str] = []

    try:
        root = ET.fromstring(raw_svg)
    except ET.ParseError as exc:
        log.error("Engine produced unparseable SVG: %s", exc)
        raise InvalidSvgError(
            "The generated vector file was malformed.",
            detail=f"XML parse error: {exc}",
        ) from exc

    removed = _cull_tiny_paths(
        root,
        canvas_width=processed_width,
        canvas_height=processed_height,
        min_ratio=params.min_path_diagonal_ratio,
    )

    _normalize_root(
        root,
        processed_width=processed_width,
        processed_height=processed_height,
        original_width=original_width,
        original_height=original_height,
    )

    svg = ET.tostring(root, encoding="unicode")

    if params.enabled:
        svg, scour_warning = _run_scour(svg, params)
        if scour_warning:
            warnings.append(scour_warning)

    path_count = validate_svg(svg)

    result = OptimizeResult(
        svg=svg,
        path_count=path_count,
        removed_paths=removed,
        bytes_before=bytes_before,
        bytes_after=len(svg.encode("utf-8")),
        warnings=warnings,
    )
    log.info(
        "SVG finalized: %d paths (%d culled), %.1f KB -> %.1f KB",
        result.path_count,
        result.removed_paths,
        result.bytes_before / 1024,
        result.bytes_after / 1024,
    )
    return result


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _normalize_root(
    root: ET.Element,
    *,
    processed_width: int,
    processed_height: int,
    original_width: int,
    original_height: int,
) -> None:
    # Do NOT set xmlns here: the element is already in the SVG namespace and
    # ET emits the declaration itself. Setting it manually produces a
    # duplicate xmlns attribute and an unparseable document.
    root.set("version", "1.1")
    # Geometry lives in processed-pixel space...
    root.set("viewBox", f"0 0 {processed_width} {processed_height}")
    # ...but the document presents itself at the source artwork's size.
    root.set("width", str(original_width))
    root.set("height", str(original_height))
    root.set("preserveAspectRatio", "xMidYMid meet")


def _path_bbox(element: ET.Element) -> tuple[float, float, float, float] | None:
    """Approximate bbox of a path from its coordinate numbers.

    Control points can sit slightly outside the true curve hull, which makes
    this a marginal over-estimate. That is the safe direction: we only ever
    keep a path we might have culled, never cull one we should have kept.
    """
    d_attr = element.get("d")
    if not d_attr:
        return None

    numbers = [float(n) for n in _NUMBER_RE.findall(d_attr)]
    if len(numbers) < 4:
        return None

    xs = numbers[0::2]
    ys = numbers[1::2]
    count = min(len(xs), len(ys))
    xs, ys = xs[:count], ys[:count]

    offset_x = offset_y = 0.0
    transform = element.get("transform")
    if transform:
        match = _TRANSLATE_RE.search(transform)
        if match:
            offset_x = float(match.group(1))
            offset_y = float(match.group(2))

    return (
        min(xs) + offset_x,
        min(ys) + offset_y,
        max(xs) + offset_x,
        max(ys) + offset_y,
    )


def _cull_tiny_paths(
    root: ET.Element, *, canvas_width: int, canvas_height: int, min_ratio: float
) -> int:
    if min_ratio <= 0:
        return 0

    canvas_diagonal = (canvas_width**2 + canvas_height**2) ** 0.5
    threshold = canvas_diagonal * min_ratio
    removed = 0

    # ElementTree has no parent pointers; walk parents explicitly.
    for parent in root.iter():
        doomed = []
        for child in list(parent):
            if not child.tag.endswith("path"):
                continue
            bbox = _path_bbox(child)
            if bbox is None:
                continue
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if (width**2 + height**2) ** 0.5 < threshold:
                doomed.append(child)
        for child in doomed:
            parent.remove(child)
            removed += 1

    return removed


def _run_scour(svg: str, params: OptimizeParams) -> tuple[str, str | None]:
    """Run scour, but never let an optimizer failure lose the artwork."""
    try:
        from scour import scour as scour_module
    except ImportError:
        return svg, "SVG optimizer unavailable; returned unoptimized output."

    options = scour_module.sanitizeOptions()
    # Significant digits, not decimal places - see OptimizeParams.
    options.digits = params.significant_digits
    options.cdigits = params.significant_digits
    options.simple_colors = True
    options.style_to_xml = True
    options.group_collapse = params.collapse_groups
    options.group_create = False
    options.strip_comments = True
    options.strip_ids = True
    options.shorten_ids = True
    options.remove_metadata = True
    options.remove_descriptive_elements = True
    options.strip_xml_prolog = False
    options.newlines = False
    options.indent_type = "none"
    options.quiet = True
    # Keep the raster embedded only if it somehow got there - validate_svg
    # rejects that case anyway, so never let scour inline new ones.
    options.embed_rasters = False
    options.enable_viewboxing = False  # we set our own viewBox above

    try:
        optimized = scour_module.scourString(svg, options)
    except Exception as exc:  # scour is strict about odd input
        log.warning("scour failed, keeping unoptimized SVG: %s", exc)
        return svg, "SVG optimization was skipped (optimizer error)."

    if "<path" not in optimized:
        log.warning("scour removed all geometry; keeping unoptimized SVG")
        return svg, "SVG optimization was skipped (it would have removed geometry)."

    return optimized, None


def validate_svg(svg: str) -> int:
    """Assert the output is genuine vector artwork. Returns the path count."""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise InvalidSvgError(
            "The generated vector file was malformed.",
            detail=f"XML parse error: {exc}",
        ) from exc

    if not root.tag.endswith("svg"):
        raise InvalidSvgError("The generated file is not an SVG document.")

    path_count = sum(1 for el in root.iter() if el.tag.endswith("path"))
    shape_count = sum(
        1
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1]
        in {"polygon", "polyline", "circle", "ellipse", "rect", "line"}
    )

    # Hard fail on faked vectorization.
    embedded = [el for el in root.iter() if el.tag.endswith("image")]
    if embedded:
        raise InvalidSvgError(
            "The generated file embedded the original raster image instead of "
            "producing vector paths."
        )
    if "data:image" in svg:
        raise InvalidSvgError(
            "The generated file contains an embedded raster data URI."
        )

    if path_count == 0 and shape_count == 0:
        raise InvalidSvgError(
            "Vectorization produced no shapes. The image may be blank, or the "
            "background removal may have erased everything - try the "
            "'detailed' preset or disable background removal."
        )

    return path_count
