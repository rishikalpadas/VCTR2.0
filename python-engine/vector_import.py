"""Import an existing vector file (PDF, Illustrator, CorelDRAW) as SVG.

Feeds the website's "vector to colour separation" flow, which only reads SVG.
The conversion is Inkscape's, driven as a subprocess the same way the Potrace
engine drives its binary: Inkscape's PDF importer is what Illustrator files
need too (every .ai saved since CS is a PDF with extra private data), and it
bundles libcdr for CorelDRAW.

The format is decided from magic bytes, never the filename:

  * ``%PDF-``             PDF, and PDF-compatible .ai
  * ``%!PS-Adobe``        legacy PostScript .ai (pre-CS) - refused, Inkscape
                          would need Ghostscript to read it
  * ``RIFF....CDR``       CorelDRAW up to X3
  * ``PK`` + ``content/`` CorelDRAW X4 and later (a zip of RIFF parts)

Only the first page of a multi-page document is converted. Text comes back as
outlines so the result does not depend on fonts the viewer may not have. A
hairline artboard border round the page is removed (see
``_drop_artboard_frames``) so it does not print on a film.

Trust boundary
--------------
The upload is untrusted, and so is Inkscape's output - it copies whatever the
document carried. Before the SVG leaves this module, scripts and
``foreignObject`` are dropped, along with ``on*`` event attributes and every
link that is not a same-document ``#id`` or an embedded ``data:image/``. The
artwork's own embedded images (a halftone screen placed as a bitmap, say)
survive: the separation renders the whole SVG to pixels, so they still get
their film.
"""

from __future__ import annotations

import io
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from config import settings
from errors import (
    CorruptImageError,
    EngineError,
    ImageTooLargeError,
    UnsupportedFormatError,
)
from logging_config import get_logger

log = get_logger(__name__)

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

# Below the website proxy's 120 s timeout, so a slow file fails with our
# message instead of the proxy's.
_INKSCAPE_TIMEOUT_S = 90

_DROPPED_TAGS = {"script", "foreignObject"}


class ImportUnavailableError(EngineError):
    code = "IMPORT_UNAVAILABLE"
    http_status = 503


def _find_inkscape() -> str | None:
    if settings.inkscape_path:
        return settings.inkscape_path if Path(settings.inkscape_path).is_file() else None
    found = shutil.which("inkscape")
    if found:
        return found
    if platform.system() == "Windows":
        for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            # inkscape.com is the console entry point: it waits for the
            # export and keeps its output, where inkscape.exe detaches.
            candidate = Path(root or "") / "Inkscape" / "bin" / "inkscape.com"
            if root and candidate.is_file():
                return str(candidate)
    return None


_VERSION_RE = re.compile(r"Inkscape (\d+)\.(\d+)")
_versions: dict[str, tuple[int, int] | None] = {}


def _inkscape_version(inkscape: str) -> tuple[int, int] | None:
    """``(major, minor)`` from ``inkscape --version``, cached per binary."""
    if inkscape not in _versions:
        try:
            proc = subprocess.run([inkscape, "--version"], capture_output=True, timeout=30)
            text = (proc.stdout + proc.stderr).decode("utf-8", "replace")
            match = _VERSION_RE.search(text)
            _versions[inkscape] = (int(match[1]), int(match[2])) if match else None
        except (OSError, subprocess.TimeoutExpired):
            _versions[inkscape] = None
        log.info("Inkscape %s -> version %s", inkscape, _versions[inkscape])
    return _versions[inkscape]


def _first_page_args(version: tuple[int, int] | None, fmt: str) -> list[str]:
    """Inkscape arguments that limit the import to the first page.

    ``--pages`` arrived with multi-page documents in Inkscape 1.3. 1.0 - 1.2
    (Debian 12 and Ubuntu 24.04 still ship 1.2.2) stop with "Unknown option
    --pages=1" before reading the file, and pick the page with the PDF
    importer's ``--pdf-page`` instead. CorelDRAW needs nothing there: 1.2
    imports only the first page anyway.
    """
    if version is None or version >= (1, 3):
        return ["--pages=1"]
    return ["--pdf-page=1"] if fmt == "pdf" else []


def detect_format(data: bytes) -> str:
    """``"pdf"`` or ``"cdr"``; raises for anything else."""
    head = data[:1024]
    if head.startswith(b"%PDF-") or b"%PDF-" in head:
        return "pdf"
    if head.startswith(b"%!PS-Adobe"):
        raise UnsupportedFormatError(
            "This is a legacy Illustrator file (saved before CS). Open it in "
            "Illustrator and save it again with \"Create PDF Compatible File\" "
            "turned on, or export it as PDF or SVG."
        )
    if data[:4] == b"RIFF" and data[8:11] in (b"CDR", b"cdr"):
        return "cdr"
    if data[:4] == b"PK\x03\x04":
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            names = []
        if any(name.lower().startswith("content/") for name in names):
            return "cdr"
    raise UnsupportedFormatError(
        "That file is not a PDF, Illustrator or CorelDRAW document."
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sanitize_svg(svg: bytes) -> tuple[str, dict]:
    """Strip anything active or external; returns the SVG and what was removed."""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise EngineError(
            "The converted file could not be read.", detail=f"XML parse error: {exc}"
        ) from exc
    if _local(root.tag) != "svg":
        raise EngineError("The converter did not return an SVG.")

    removed = {"elements": 0, "attributes": 0, "links": 0}
    for parent in list(root.iter()):
        for child in list(parent):
            if _local(child.tag) in _DROPPED_TAGS:
                parent.remove(child)
                removed["elements"] += 1

    for element in root.iter():
        for name in list(element.attrib):
            if _local(name).lower().startswith("on"):
                del element.attrib[name]
                removed["attributes"] += 1
        for name in ("href", _XLINK_HREF):
            value = (element.get(name) or "").strip()
            if value and not (value.startswith("#") or value.startswith("data:image/")):
                del element.attrib[name]
                removed["links"] += 1

    removed["frames"] = _drop_artboard_frames(root)

    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    return ET.tostring(root, encoding="unicode"), removed


# ---------------------------------------------------------------------------
# Artboard frames
# ---------------------------------------------------------------------------
#
# Illustrator and print workflows often leave a hairline box round the whole
# page - an artboard or trim border, typically a 0.2 pt black stroke. It is not
# artwork, but to a colour separation it is ink: it would land on the black
# film (or get a film of its own). An unfilled, axis-aligned rectangle with a
# sub-pixel stroke that spans most of the canvas is that box, and is dropped.

_TRANSFORM_RE = re.compile(r"(matrix|translate|scale)\s*\(([^)]*)\)")
_PATH_TOKEN_RE = re.compile(r"[MLHVZmlhvz]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_FRAME_MAX_STROKE = 1.0  # user units, after transforms
_FRAME_MIN_SPAN = 0.5  # of the canvas width and height


def _parse_transform(value: str | None):
    """Affine (a, b, c, d, e, f) for the forms Inkscape writes, else None."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not value:
        return m
    rest = _TRANSFORM_RE.sub("", value).strip(" ,")
    if rest:
        return None  # rotate/skew or something unparsed: leave the element be
    for kind, args in _TRANSFORM_RE.findall(value):
        n = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if kind == "matrix" and len(n) == 6:
            t = tuple(n)
        elif kind == "translate" and n:
            t = (1, 0, 0, 1, n[0], n[1] if len(n) > 1 else 0)
        elif kind == "scale" and n:
            t = (n[0], 0, 0, n[1] if len(n) > 1 else n[0], 0, 0)
        else:
            return None
        a, b, c, d, e, f = m
        A, B, C, D, E, F = t
        m = (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D,
             a * E + c * F + e, b * E + d * F + f)
    return m


def _rect_corners(element: ET.Element):
    """The rectangle's corners, or None if the element is not a plain box."""
    tag = _local(element.tag)
    if tag == "rect":
        try:
            x, y = float(element.get("x", 0)), float(element.get("y", 0))
            w, h = float(element.get("width", 0)), float(element.get("height", 0))
        except ValueError:
            return None
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if tag != "path":
        return None

    tokens = _PATH_TOKEN_RE.findall(element.get("d") or "")
    points, x, y, cmd, i = [], 0.0, 0.0, None, 0
    try:
        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha():
                cmd = tok
                i += 1
                if cmd in "Zz":
                    continue
                if cmd.islower():
                    return None  # relative commands: not worth the risk
                continue
            if cmd in ("M", "L"):
                x, y = float(tokens[i]), float(tokens[i + 1])
                i += 2
            elif cmd == "H":
                x = float(tok)
                i += 1
            elif cmd == "V":
                y = float(tok)
                i += 1
            else:
                return None
            points.append((x, y))
    except (IndexError, ValueError):
        return None

    if points and points[-1] == points[0]:
        points.pop()
    xs, ys = {p[0] for p in points}, {p[1] for p in points}
    if len(points) != 4 or len(xs) != 2 or len(ys) != 2:
        return None
    return points


def _canvas_size(root: ET.Element):
    box = [float(v) for v in re.split(r"[\s,]+", (root.get("viewBox") or "").strip()) if v]
    if len(box) == 4 and box[2] > 0 and box[3] > 0:
        return box[2], box[3]
    try:
        return float(root.get("width", "").rstrip("px")), float(root.get("height", "").rstrip("px"))
    except ValueError:
        return None


def _drop_artboard_frames(root: ET.Element) -> int:
    canvas = _canvas_size(root)
    if not canvas:
        return 0
    canvas_w, canvas_h = canvas
    dropped = 0

    def walk(parent, matrix):
        nonlocal dropped
        for element in list(parent):
            if _local(element.tag) in ("defs", "clipPath", "mask", "pattern", "symbol"):
                continue
            own = _parse_transform(element.get("transform"))
            if own is None or matrix is None:
                combined = None
            else:
                a, b, c, d, e, f = matrix
                A, B, C, D, E, F = own
                combined = (a * A + c * B, b * A + d * B, a * C + c * D,
                            b * C + d * D, a * E + c * F + e, b * E + d * F + f)
            if combined is not None and _is_frame(element, combined, canvas_w, canvas_h):
                parent.remove(element)
                dropped += 1
                continue
            walk(element, combined)

    walk(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    return dropped


def _is_frame(element, matrix, canvas_w, canvas_h) -> bool:
    props = {}
    for item in (element.get("style") or "").split(";"):
        key, sep, value = item.partition(":")
        if sep:
            props[key.strip()] = value.strip()
    fill = props.get("fill", element.get("fill"))
    stroke = props.get("stroke", element.get("stroke"))
    if fill != "none" or not stroke or stroke == "none":
        return False

    corners = _rect_corners(element)
    if not corners:
        return False
    a, b, c, d, e, f = matrix
    if abs(b) > 1e-9 or abs(c) > 1e-9:
        return False  # rotated: not an artboard box
    xs = [a * x + e for x, _ in corners]
    ys = [d * y + f for _, y in corners]
    if (max(xs) - min(xs)) < _FRAME_MIN_SPAN * canvas_w:
        return False
    if (max(ys) - min(ys)) < _FRAME_MIN_SPAN * canvas_h:
        return False

    try:
        width = float(re.sub(r"[a-z%]+$", "", props.get("stroke-width", element.get("stroke-width", "1"))))
    except ValueError:
        return False
    return width * (abs(a * d) ** 0.5) <= _FRAME_MAX_STROKE


def import_vector_file(data: bytes, filename: str = "") -> tuple[str, dict]:
    """Convert a PDF / AI / CDR upload to sanitized SVG text and metadata."""
    if not data:
        raise CorruptImageError("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"File is larger than the "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB limit."
        )

    fmt = detect_format(data)
    inkscape = _find_inkscape()
    if not inkscape:
        raise ImportUnavailableError(
            "PDF, AI and CDR import is not available on this server.",
            detail="Inkscape not found; install it or set VEC_INKSCAPE_PATH",
        )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vec-import-") as tmp:
        # Inkscape picks its importer from the extension, so it is set from the
        # detected format - the uploaded name is never used on disk.
        source = Path(tmp) / f"input.{fmt}"
        target = Path(tmp) / "output.svg"
        source.write_bytes(data)
        command = [
            inkscape,
            *_first_page_args(_inkscape_version(inkscape), fmt),
            "--export-type=svg",
            "--export-plain-svg",
            "--export-text-to-path",
            f"--export-filename={target}",
            str(source),
        ]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                timeout=_INKSCAPE_TIMEOUT_S,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineError(
                "Converting this file took too long. Try a simpler file.",
                detail=f"inkscape exceeded {_INKSCAPE_TIMEOUT_S}s",
            ) from exc

        if not target.is_file() or target.stat().st_size == 0:
            stderr = proc.stderr.decode("utf-8", "replace")[-500:]
            raise CorruptImageError(
                "This file could not be converted. It may be damaged, "
                "password-protected or saved in an unsupported version.",
                detail=f"inkscape exit {proc.returncode}: {stderr}",
            )
        svg, removed = sanitize_svg(target.read_bytes())

    elapsed = (time.perf_counter() - started) * 1000
    meta = {
        "format": fmt,
        "source_bytes": len(data),
        "svg_bytes": len(svg.encode("utf-8")),
        "first_page_only": True,
        "removed": removed,
        "elapsed_ms": round(elapsed),
    }
    log.info(
        "Imported %s (%s, %.1f KB) -> %.1f KB SVG in %.0f ms; removed %s",
        fmt,
        filename[:80] or "upload",
        len(data) / 1024,
        meta["svg_bytes"] / 1024,
        elapsed,
        removed,
    )
    return svg, meta
