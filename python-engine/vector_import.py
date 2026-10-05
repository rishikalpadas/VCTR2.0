"""Vector-file import: PDF / Adobe Illustrator (.ai) / CorelDRAW (.cdr) -> SVG.

Used by the colour-separation flow, which only understands SVG. All real work
is done by the Inkscape CLI (PDF import via poppler, CDR via libcdr), run as an
untrusted-input subprocess: private temp dir, throwaway HOME, hard timeout.

Format detection is by magic bytes, never by filename:

* ``%PDF-``                 PDF - and every modern .ai, which is a PDF with
                            Illustrator's private data appended.
* ``%!PS-Adobe`` (+ Illustrator marker)
                            legacy .ai (PostScript). Inkscape cannot read it;
                            we say so rather than fail obscurely.
* ``RIFF....CDR`` / ``PK``  CorelDRAW (X3 and earlier are RIFF, X4+ is a ZIP).

The SVG that comes back is sanitised before it leaves the engine, because it
is derived from an attacker-controlled file and will be rendered by a browser.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from errors import VectorizationError
from logging_config import get_logger

log = get_logger("vector_import")

INKSCAPE_TIMEOUT_SECONDS = 90
MAX_SVG_BYTES = 40 * 1024 * 1024

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_BANNED_TAGS = {"script", "foreignobject", "iframe", "audio", "video"}
_SAFE_HREF_RE = re.compile(r"^(#|data:image/(png|jpeg|jpg|gif|webp);base64,)", re.IGNORECASE)
_DOCTYPE_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


class UnsupportedVectorFormatError(VectorizationError):
    code = "UNSUPPORTED_VECTOR_FORMAT"
    http_status = 415


class VectorImportError(VectorizationError):
    code = "VECTOR_IMPORT_FAILED"
    http_status = 422


class ImporterUnavailableError(VectorizationError):
    code = "IMPORTER_UNAVAILABLE"
    http_status = 503


@dataclass(frozen=True)
class ImportOutcome:
    svg: str
    meta: dict


def _find_inkscape() -> str | None:
    configured = os.environ.get("VEC_INKSCAPE_BIN")
    if configured and Path(configured).exists():
        return configured
    return shutil.which("inkscape") or shutil.which("inkscape.com")


def importer_available() -> bool:
    return _find_inkscape() is not None


def detect_format(data: bytes) -> str:
    """Return ``pdf``, ``cdr`` or raise UnsupportedVectorFormatError."""
    head = data[:2048]
    if head.startswith(b"%PDF-") or b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(b"RIFF") and head[8:11] == b"CDR":
        return "cdr"
    if head.startswith(b"PK\x03\x04") and b"content/riffData.cdr" in data[:4096] + data[-4096:]:
        return "cdr"
    if head.startswith(b"%!PS-Adobe") or head.startswith(b"\x04%!PS-Adobe"):
        raise UnsupportedVectorFormatError(
            "This is an older Illustrator file saved without PDF compatibility. "
            "Re-save it from Illustrator with 'Create PDF Compatible File' "
            "ticked, or export it as PDF or SVG."
        )
    raise UnsupportedVectorFormatError(
        "Unrecognised file. Upload a PDF, Illustrator (.ai) or CorelDRAW (.cdr) file."
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _sanitise(svg_text: str) -> tuple[str, dict]:
    """Strip anything executable or fetch-capable from imported SVG."""
    if _DOCTYPE_RE.search(svg_text):
        svg_text = _DOCTYPE_RE.sub("<!-- removed -->", svg_text)
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise VectorImportError("The converted file could not be parsed.", detail=str(exc)) from exc
    if _local(root.tag) != "svg":
        raise VectorImportError("The converter did not produce an SVG.")

    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", _XLINK_NS)

    drawables = 0
    images = 0
    for parent in list(root.iter()):
        for child in list(parent):
            if _local(child.tag) in _BANNED_TAGS:
                parent.remove(child)
    for el in root.iter():
        name = _local(el.tag)
        if name in {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line"}:
            drawables += 1
        elif name == "image":
            images += 1
        for attr in list(el.attrib):
            local_attr = attr.rsplit("}", 1)[-1].lower()
            if local_attr.startswith("on"):
                del el.attrib[attr]
            elif local_attr == "href" and not _SAFE_HREF_RE.match(el.attrib[attr].strip()):
                del el.attrib[attr]

    if drawables == 0 and images == 0:
        raise VectorImportError("The file converted but contains no visible artwork.")

    out = ET.tostring(root, encoding="unicode")
    if len(out.encode("utf-8")) > MAX_SVG_BYTES:
        raise VectorImportError("The converted artwork is too large to process.")
    return out, {"shapes": drawables, "embedded_images": images}


def import_to_svg(data: bytes) -> ImportOutcome:
    fmt = detect_format(data)
    binary = _find_inkscape()
    if binary is None:
        raise ImporterUnavailableError(
            "File import is not available on this server right now.",
            detail="inkscape binary not found (install it or set VEC_INKSCAPE_BIN)",
        )

    with tempfile.TemporaryDirectory(prefix="vimport_") as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / f"input.{fmt}"
        dst = tmp_dir / "output.svg"
        src.write_bytes(data)
        home = tmp_dir / "home"
        home.mkdir()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        }
        cmd = [
            binary,
            str(src),
            "--export-type=svg",
            "--export-plain-svg",
            "--export-text-to-path",
            f"--export-filename={dst}",
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=tmp, env=env, capture_output=True,
                timeout=INKSCAPE_TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VectorImportError(
                "Converting this file took too long. Try a simpler file."
            ) from exc

        if proc.returncode != 0 or not dst.exists():
            stderr = proc.stderr.decode("utf-8", "replace")[-800:]
            log.warning("inkscape failed (%s, rc=%s): %s", fmt, proc.returncode, stderr)
            hint = (
                "This CorelDRAW file could not be read - newer .cdr versions are "
                "often unsupported. Export it from CorelDRAW as SVG or PDF."
                if fmt == "cdr"
                else "This file could not be read. Re-save it as PDF or SVG and try again."
            )
            raise VectorImportError(hint, detail=stderr)

        svg, stats = _sanitise(dst.read_text(encoding="utf-8", errors="replace"))

    log.info("Imported %s -> SVG (%d shapes, %d images)", fmt, stats["shapes"], stats["embedded_images"])
    return ImportOutcome(svg=svg, meta={"source_format": fmt, **stats})
