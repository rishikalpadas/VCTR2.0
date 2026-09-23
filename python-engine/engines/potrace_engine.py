"""Potrace engine.

Potrace (Peter Selinger) is the reference C tracer most other "clean vector
outline" tools either wrap or imitate. Unlike VTracer it only ever traces one
colour (black ink on white) per run, so this engine does the colour-layer
splitting itself: it builds one binary mask per surviving colour in the
already-quantized/preprocessed image and calls the potrace binary once per
mask, then merges the per-layer paths back into a single SVG.

Why shell out to the real C binary instead of a Python potrace binding: the
only pure-Python port on PyPI (``potracer``) was checked against known shapes
(a plain rectangle, a filled circle) during development and returned wrong
geometry - corner-connecting midpoint diamonds instead of the actual outline -
regardless of input convention. The real binary is vendored instead (see
``vendor/potrace/``, downloaded from potrace.sourceforge.net and verified
against the project's published SHA1 sum) and driven as a subprocess.

Potrace's own SVG backend emits path data in an internal coordinate system
(y-up, scaled by 1/unit) wrapped in a single ``<g transform="translate(...)
scale(...)">``. Rather than carrying that transform through to the merged
document - which ``svg_optimizer``'s tiny-path culler cannot interpret, since
it only understands a plain ``translate()`` - this engine bakes the transform
into each path's coordinates itself and emits plain absolute ``<path d="...">
fill="#hex"/>`` elements.
"""

from __future__ import annotations

import platform
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from errors import EngineError
from logging_config import get_logger
from presets import Preset, PotraceParams

from .base import BaseVectorizer, EngineResult, registry

log = get_logger(__name__)

_BIN_DIR = Path(__file__).resolve().parent.parent / "vendor" / "potrace"
# The vendored binary is platform-specific (downloaded from
# potrace.sourceforge.net's own precompiled distributions - see the
# module docstring). "potrace.exe" only exists on the Windows dev box this
# was built on; a Linux/macOS deployment needs the matching ELF/Mach-O
# binary dropped in next to it, named plain "potrace".
_POTRACE_EXE = _BIN_DIR / ("potrace.exe" if platform.system() == "Windows" else "potrace")

_SVG_NS = "http://www.w3.org/2000/svg"
_GROUP_TRANSFORM_RE = re.compile(
    r"translate\(\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)\s*\)\s*"
    r"scale\(\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)\s*\)"
)
_TOKEN_RE = re.compile(r"([MmLlCcZz])([^MmLlCcZz]*)")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

_MIN_LAYER_PIXELS = 4
_SUBPROCESS_TIMEOUT_S = 20


class PotraceVectorizer(BaseVectorizer):
    name = "potrace"
    description = (
        "Potrace 1.16 (Peter Selinger) - the reference C tracer, wrapped with "
        "a colour-layer splitter. Least-squares corner/curve fitting tends to "
        "read cleaner than spline fitting on flat art and line work; one "
        "subprocess call per colour makes it slower on busy, unquantized art."
    )

    def is_available(self) -> bool:
        return _POTRACE_EXE.exists()

    def vectorize(self, rgba: np.ndarray, preset: Preset) -> EngineResult:
        if not self.is_available():
            raise EngineError(
                "The Potrace engine is not available on this server.",
                detail=f"binary not found at {_POTRACE_EXE}",
            )

        params = preset.engine_params
        if not isinstance(params, PotraceParams):
            params = PotraceParams()

        started = time.perf_counter()
        height, width = rgba.shape[:2]

        skip_background = preset.preprocess.binarize
        layers = _extract_color_layers(
            rgba, max_layers=params.max_layers, skip_lightest=skip_background
        )
        layers = _tuck_fills_under_ink(
            layers, reach=params.ink_tuck, speck_area=params.speck_area
        )

        if not layers:
            raise EngineError(
                "The vectorization engine returned no usable output.",
                detail="no opaque colour layers survived extraction",
            )

        path_elements: list[str] = []
        layer_meta: list[dict] = []

        with tempfile.TemporaryDirectory(prefix="potrace_") as tmp:
            tmp_dir = Path(tmp)
            for i, (hex_color, mask) in enumerate(layers):
                pgm_path = tmp_dir / f"layer_{i}.pgm"
                svg_path = tmp_dir / f"layer_{i}.svg"
                _write_pgm(pgm_path, mask)

                cmd = [
                    str(_POTRACE_EXE),
                    "-s",
                    "-o", str(svg_path),
                    *params.as_cli_args(),
                    str(pgm_path),
                ]
                try:
                    subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=_SUBPROCESS_TIMEOUT_S,
                        check=True,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise EngineError(
                        "Vectorization timed out while tracing a colour layer.",
                        detail=str(exc),
                    ) from exc
                except subprocess.CalledProcessError as exc:
                    raise EngineError(
                        "Vectorization failed while tracing a colour layer.",
                        detail=(exc.stderr or b"").decode("utf-8", "replace"),
                    ) from exc

                if not svg_path.exists():
                    continue  # nothing traced for this layer (e.g. fully filtered by turdsize)

                ds = _extract_absolute_path_ds(svg_path.read_text(encoding="utf-8"))
                for d in ds:
                    path_elements.append(f'<path fill="{hex_color}" d="{d}"/>')
                if ds:
                    layer_meta.append({"color": hex_color, "subpaths": len(ds)})

        if not path_elements:
            raise EngineError(
                "The vectorization engine returned no usable output.",
                detail="every colour layer traced to an empty path",
            )

        elapsed_ms = (time.perf_counter() - started) * 1000

        svg = (
            f'<svg xmlns="{_SVG_NS}" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            + "".join(path_elements)
            + "</svg>"
        )

        log.info(
            "Potrace traced %dx%d in %.0f ms (%d layers, %d paths, %.1f KB)",
            width,
            height,
            elapsed_ms,
            len(layer_meta),
            len(path_elements),
            len(svg) / 1024,
        )

        return EngineResult(
            svg=svg,
            engine=self.name,
            meta={
                "engine_ms": round(elapsed_ms, 1),
                "raw_path_count": len(path_elements),
                "raw_svg_bytes": len(svg.encode("utf-8")),
                "layer_count": len(layer_meta),
                "layers": layer_meta,
                "params": {
                    "turnpolicy": params.turnpolicy,
                    "turdsize": params.turdsize,
                    "alphamax": params.alphamax,
                    "opttolerance": params.opttolerance,
                },
            },
        )


# ---------------------------------------------------------------------------
# Colour-layer extraction
# ---------------------------------------------------------------------------


def _extract_color_layers(
    rgba: np.ndarray, *, max_layers: int, skip_lightest: bool
) -> list[tuple[str, np.ndarray]]:
    """Split an RGBA image into (hex colour, binary mask) layers, largest first.

    Fully transparent pixels never form a layer - they stay untraced, same as
    a background-removed image handed to VTracer. If more distinct colours
    survive than ``max_layers`` (art that reached this engine without upstream
    quantization), they are collapsed with k-means so this never turns into
    hundreds of subprocess calls.
    """
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else np.full((h, w), 255, dtype=np.uint8)
    opaque = alpha >= 128
    if not np.any(opaque):
        return []

    flat_rgb = rgb.reshape(-1, 3)
    opaque_flat = opaque.reshape(-1)
    opaque_rgb = flat_rgb[opaque_flat]

    colors, inverse, counts = np.unique(
        opaque_rgb, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)

    if len(colors) > max_layers:
        colors, inverse, counts = _requantize(opaque_rgb, max_layers)

    label = np.full(h * w, -1, dtype=np.int32)
    label[opaque_flat] = inverse
    label = label.reshape(h, w)

    order = np.argsort(-counts)

    if skip_lightest:
        # Binarized (line-art style) input: the paler colour is the flattened
        # white canvas potrace's own convention would leave untraced, not
        # real artwork - see preprocessing._binarize.
        brightness = colors.sum(axis=1)
        lightest = int(np.argmax(brightness))
        order = [i for i in order if i != lightest]

    order = _ink_on_top(order, colors)

    layers = []
    for idx in order:
        idx = int(idx)
        count = int(counts[idx])
        if count < _MIN_LAYER_PIXELS:
            continue
        mask = label == idx
        hex_color = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in colors[idx]])
        layers.append((hex_color, mask))
    return layers


def _ink_on_top(order, colors: np.ndarray) -> list[int]:
    """Move the darkest colour to the end of the paint order.

    Every layer is traced independently, so the boundary two regions share
    comes back as two slightly different curves. Wherever they disagree the
    layer painted later wins the seam. With the darkest layer painted first -
    which is where sorting by area puts it, since on a dark background it is
    also the largest - every fill in the artwork gets to bulge into the line
    work by whatever fraction of a pixel its own curve fit happened to land
    on. The stroke then reads thinner in some places than others, and the fill
    that ate it shows up as colour inside the stroke corridor. Both are the
    same defect, and both are what "the dark lines are not consistent" looks
    like.

    Painting the ink last is also what the artwork means: in flat vector art
    the line work sits on top. It is safe to reorder because the masks are
    disjoint by construction - a layer can only ever win a seam a pixel wide,
    never cover another region.
    """
    if len(order) < 2:
        return [int(i) for i in order]
    luma = colors.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    darkest = int(np.argmin(luma))
    return [int(i) for i in order if int(i) != darkest] + [darkest]


def _tuck_fills_under_ink(
    layers: list[tuple[str, np.ndarray]], *, reach: float, speck_area: int
) -> list[tuple[str, np.ndarray]]:
    """Run every fill on underneath the line work, and fold specks away.

    ``layers`` is in paint order with the ink last (see ``_ink_on_top``).

    Every layer is traced on its own mask, so a fill that stops exactly at the
    ink comes back with its outline on the stroke's edge - and wherever its
    curve fit bulges by a fraction of a pixel, the fill pokes out past the
    stroke into the next region, or falls short and leaves a background
    sliver. Both read as colour leaking across the line work. Handing each ink
    pixel to its nearest fill moves every fill outline to the *middle* of the
    stroke instead, where the ink painted on top covers whatever the fit does.
    The ink's own mask is untouched, so the strokes keep their traced weight.

    ``reach`` bounds how far a fill runs under ink. Past a stroke's half width
    the extra area is invisible anyway, and without a bound a dark backdrop
    that quantized onto the ink colour would turn the neighbouring fill into
    a canvas-sized shape. Transparent pixels take part as their own seed, so a
    stroke against a removed background is split with nothing rather than
    growing the fill out past the artwork.

    Specks - fill components under ``speck_area`` pixels - are merged into the
    nearest other fill first. At three-colour junctions quantization strands
    a few pixels of the wrong colour against the stroke; potrace's own
    turdsize filter only drops them, which leaves a hole of background.
    """
    if len(layers) < 2 or reach <= 0:
        return layers

    height, width = layers[0][1].shape
    ink_index = len(layers) - 1
    label = np.full((height, width), -1, dtype=np.int32)  # -1: transparent
    for index, (_, mask) in enumerate(layers):
        label[mask] = index

    if speck_area > 0:
        for index in range(ink_index):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (label == index).astype(np.uint8), connectivity=8
            )
            small = [c for c in range(1, count) if stats[c, cv2.CC_STAT_AREA] < speck_area]
            if not small:
                continue
            speck = np.isin(components, small)
            others = (label >= 0) & (label != ink_index) & (label != index)
            if not others.any():
                continue
            nearest, _ = _nearest_seed(others, label)
            label[speck] = nearest[speck]

    ink = label == ink_index
    nearest, distance = _nearest_seed(~ink, label)
    tucked = ink & (distance <= reach) & (nearest >= 0)

    result = []
    for index, (hex_color, mask) in enumerate(layers[:-1]):
        grown = (label == index) | (tucked & (nearest == index))
        if int(grown.sum()) >= _MIN_LAYER_PIXELS:
            result.append((hex_color, grown))
    result.append((layers[-1][0], ink))
    return result


def _nearest_seed(
    seeds: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel, ``values`` at the nearest seed pixel and its distance."""
    distance, labels = cv2.distanceTransformWithLabels(
        np.where(seeds, 0, 1).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    # DIST_LABEL_PIXEL numbers the seed pixels 1..n in raster order.
    ys, xs = np.nonzero(seeds)
    lookup = np.concatenate([[-1], values[ys, xs]]).astype(values.dtype)
    return lookup[labels], distance


def _requantize(
    samples: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = samples.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(-1)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    counts = np.bincount(labels, minlength=k)
    return centers, labels, counts


def _write_pgm(path: Path, mask: np.ndarray) -> None:
    """Write a binary mask as a raw PGM (foreground = black = 0)."""
    h, w = mask.shape
    image = np.where(mask, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(image.tobytes())


# ---------------------------------------------------------------------------
# Potrace SVG -> absolute-coordinate path data
# ---------------------------------------------------------------------------


def _extract_absolute_path_ds(svg_text: str) -> list[str]:
    root = ET.fromstring(svg_text)
    out: list[str] = []
    for g in root.iter(f"{{{_SVG_NS}}}g"):
        match = _GROUP_TRANSFORM_RE.search(g.get("transform", ""))
        if match:
            tx, ty, sx, sy = (float(v) for v in match.groups())
        else:
            tx, ty, sx, sy = 0.0, 0.0, 1.0, 1.0
        for path_el in g.iter(f"{{{_SVG_NS}}}path"):
            d = path_el.get("d")
            if d:
                out.append(_transform_path_d(d, tx, ty, sx, sy))
    return out


def _transform_path_d(d: str, tx: float, ty: float, sx: float, sy: float) -> str:
    """Re-emit a potrace path ``d`` string as absolute, already-transformed
    coordinates. Potrace's SVG backend only ever emits M (absolute, once per
    subpath), l/c (relative, possibly repeating) and z - see potrace's
    ``backend_svg.c``. Handled generically regardless.
    """

    def tf(x: float, y: float) -> tuple[float, float]:
        return (sx * x + tx, sy * y + ty)

    parts: list[str] = []
    cur = (0.0, 0.0)
    subpath_start = (0.0, 0.0)

    for cmd, arg_str in _TOKEN_RE.findall(d):
        nums = [float(n) for n in _NUM_RE.findall(arg_str)]

        if cmd in "Mm":
            for i in range(0, len(nums), 2):
                dx, dy = nums[i], nums[i + 1]
                pt = (dx, dy) if cmd == "M" else (cur[0] + dx, cur[1] + dy)
                cur = pt
                if i == 0:
                    subpath_start = pt
                x, y = tf(*pt)
                parts.append(f"{'M' if i == 0 else 'L'} {x:.3f},{y:.3f}")
        elif cmd in "Ll":
            for i in range(0, len(nums), 2):
                dx, dy = nums[i], nums[i + 1]
                pt = (dx, dy) if cmd == "L" else (cur[0] + dx, cur[1] + dy)
                cur = pt
                x, y = tf(*pt)
                parts.append(f"L {x:.3f},{y:.3f}")
        elif cmd in "Cc":
            for i in range(0, len(nums), 6):
                base = cur
                if cmd == "C":
                    c1 = (nums[i], nums[i + 1])
                    c2 = (nums[i + 2], nums[i + 3])
                    end = (nums[i + 4], nums[i + 5])
                else:
                    c1 = (base[0] + nums[i], base[1] + nums[i + 1])
                    c2 = (base[0] + nums[i + 2], base[1] + nums[i + 3])
                    end = (base[0] + nums[i + 4], base[1] + nums[i + 5])
                cur = end
                x1, y1 = tf(*c1)
                x2, y2 = tf(*c2)
                xe, ye = tf(*end)
                parts.append(
                    f"C {x1:.3f},{y1:.3f} {x2:.3f},{y2:.3f} {xe:.3f},{ye:.3f}"
                )
        elif cmd in "Zz":
            parts.append("Z")
            cur = subpath_start

    return " ".join(parts)


registry.register(PotraceVectorizer())
