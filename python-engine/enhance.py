"""AI upscale + restore of soft, low-resolution uploads before tracing.

Customers upload what they have: a logo saved off a chat app, a screenshot of
a screenshot, a 600px JPEG that was blown up blockily and recompressed. Every
later stage reads the pixels literally. Quantization spends palette entries on
the blur ramps, the majority vote rounds what blur already rounded, and
Potrace faithfully traces the staircase a blocky upscale left on every
diagonal. Bicubic supersampling cannot help: it enlarges the blur.

This module rebuilds the image with Real-ESRGAN (compact SRVGG models, BSD-3,
exported to ONNX by ``tools/export_sr_models.py`` and run with OpenCV's own dnn
module - no extra dependency) before the pipeline sees it:

  1. measure how blurred the edges are, and from that the resolution the
     image *actually* holds - a blurry 1600px upload may carry 600px of detail;
  2. reduce it to that resolution. This is the step that makes it work: at its
     true resolution a blocky diagonal is a clean one-pixel aliased line, which
     the model rebuilds as a smooth edge. Fed at its pixel size, the model
     reads the staircase as real detail and sharpens it;
  3. run the model (4x);
  4. pull the colours back to the source. The model may shift a flat colour
     slightly, and colour fidelity is a hard requirement, so the low-frequency
     difference to the source is added back. Only colour moves - edges stay
     where the model put them.

Sharp images with enough resolution are left alone: the gate is in
``_should_enhance``. VEC_ENHANCE=false turns the whole stage off, and a request
can force it either way with the ``enhance`` option.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import settings
from logging_config import get_logger

log = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models" / "sr"
# Flat artwork - logos, lettering, stickers, cartoons. Trained on animation
# frames: rebuilds hard edges and flat fills and invents no texture, which is
# what tracing wants.
FLAT_MODEL = "realesr-animevideov3"
# Photographs and photo-like art. It restores texture, which on flat art
# reads as grain on every edge, so it is kept for the inputs that have some.
PHOTO_MODEL = "realesr-general-x4v3"
MODEL_SCALE = 4

ENHANCE_MODES = ("auto", "always", "never")

# Longest edge handed to the model. Its output is 4x this, and the tracer
# works at up to settings.absolute_max_dimension (3000px), so a bigger input
# only costs time. ~1.2s on the flat model at this size on an 8-core CPU.
_MAX_MODEL_INPUT = 768
# Tile size for inference, bounding memory (the model keeps 64 feature
# channels at full input resolution). Tiles overlap so their seams can be cut
# away from the borders the model saw zero-padded.
_TILE = 384
_TILE_PAD = 16

# Edge-sharpness ratio -> total Gaussian blur of an edge, in pixels. The ratio
# is the 90th percentile, over strong edges, of the Sobel magnitude divided by
# the local contrast: a perfectly hard step scores 4. Calibrated by blurring
# four sharp test images by known amounts - all four agreed within 0.1 at
# every point. A clean anti-aliased edge already measures about 0.4-0.5.
_SHARPNESS_TO_SIGMA = (
    (1.38, 4.0),
    (1.48, 3.0),
    (1.55, 2.5),
    (1.70, 2.0),
    (1.90, 1.65),
    (2.20, 1.35),
    (2.55, 1.1),
    (2.88, 0.9),
    (3.55, 0.65),
    (3.95, 0.45),
)
# Blur of an edge that is as sharp as its pixel grid allows.
_SHARP_SIGMA = 0.5
# How much of the measured blur factor is spent on reducing the image before
# the model. The measure slightly overstates it (JPEG softening adds to the
# blur but not to lost resolution), and reducing too far erases line work the
# model could have kept. Measured over 18 degraded test images: 0.8 cut edge
# error on blurry uploads by up to 46% against the full factor (sample_flat_art
# 39.2 -> 21.0) at no cost on blocky ones.
_REDUCE_BIAS = 0.8
# Local contrast that makes a pixel part of a "strong edge" for the measure.
_EDGE_CONTRAST = 60.0
# Fewer strong-edge pixels than this and there is nothing to measure.
_MIN_EDGE_PIXELS = 400

# The gate. Enhance when the image holds less detail than the tracer can use
# - blurred edges or a small canvas - and leave it alone otherwise.
#   * blur factor >= 1.4 (edge blur >= 0.7px): edges at least 1.4x wider
#     than sharp. Clean digital art measures 1.0-1.35; the soft and blocky
#     uploads in the test set 1.9-3.0. Between the two (1.4-1.6) sit
#     slightly soft JPEGs, where enhancing was checked by eye and helped or
#     made no difference - a watercolour sticker's broken outlines came back
#     whole. About half of 218 real customer uploads pass the gate.
#   * effective size < 700px: even sharp, a small image gives Potrace too few
#     pixels per curve; the model's 4x is worth more than bicubic's.
_MIN_BLUR_FACTOR = 1.4
_SMALL_IMAGE = 700
# Above this much real detail the tracer is not short of resolution, however
# soft the edges look (a very large photo with shallow focus, say).
_ENOUGH_DETAIL = 1400

# Colour restoration: blur radius, in model-input pixels, of the low-frequency
# colour difference that is added back. Wide enough that the difference
# between a soft source edge and the model's sharp one averages out, narrow
# enough to follow a colour that changes from one region to the next.
_COLOR_SIGMA = 4.0

# Thin-line detection for the anti-ringing clamp, in model-input pixels: a
# dark (or light) structure narrower than this kernel standing out by more
# than _LINE_CONTRAST (of the 0-1 range) counts as a line.
_LINE_KERNEL = 5
_LINE_CONTRAST = 0.08


@dataclass
class EnhanceResult:
    image: np.ndarray  # RGBA uint8
    applied: bool
    report: dict


_lock = threading.Lock()
_nets: dict[str, object] = {}


def parse_mode(value) -> str:
    """Request option -> mode. Accepts the mode names and booleans."""
    if value is True or value in ("true", "on"):
        return "always"
    if value is False or value in ("false", "off"):
        return "never"
    return value if value in ENHANCE_MODES else "auto"


def available() -> bool:
    return (MODELS_DIR / f"{FLAT_MODEL}.onnx").exists()


def _net(name: str):
    """Load a model once per process. None if missing or unreadable (cached)."""
    if name not in _nets:
        path = MODELS_DIR / f"{name}.onnx"
        try:
            _nets[name] = cv2.dnn.readNetFromONNX(str(path)) if path.exists() else None
        except cv2.error as exc:
            log.warning("Super-resolution model %s unavailable: %s", name, exc)
            _nets[name] = None
    return _nets[name]


def measure_blur(rgba: np.ndarray) -> dict:
    """Estimate edge blur and the resolution the image really holds."""
    height, width = rgba.shape[:2]
    rgb = rgba[..., :3]
    if rgba.shape[2] == 4:
        # Judge transparent art on white, as it will be seen.
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        rgb = rgb.astype(np.float32) * alpha + 255 * (1 - alpha)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    kernel = np.ones((7, 7), np.uint8)
    contrast = cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)
    edges = contrast > _EDGE_CONTRAST
    longest = max(width, height)
    if int(edges.sum()) < _MIN_EDGE_PIXELS:
        return {"sharpness": None, "blur_sigma": None, "blur_factor": 1.0,
                "effective_size": longest}

    magnitude = np.hypot(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    )
    sharpness = float(np.percentile(magnitude[edges] / (contrast[edges] + 1), 90))
    ratios, sigmas = zip(*_SHARPNESS_TO_SIGMA)
    sigma = float(np.interp(sharpness, ratios, sigmas))
    factor = float(np.clip(sigma / _SHARP_SIGMA, 1.0, 4.0))
    return {
        "sharpness": round(sharpness, 3),
        "blur_sigma": round(sigma, 2),
        "blur_factor": round(factor, 2),
        "effective_size": int(round(longest / factor)),
    }


def _should_enhance(blur: dict) -> tuple[bool, str]:
    effective = blur["effective_size"]
    if effective >= _ENOUGH_DETAIL:
        return False, f"enough detail already ({effective}px effective)"
    if blur["blur_factor"] >= _MIN_BLUR_FACTOR:
        return True, f"soft edges (blur {blur['blur_sigma']}px)"
    if effective < _SMALL_IMAGE:
        return True, f"small image ({effective}px)"
    return False, "sharp, with enough resolution"


def enhance(rgba: np.ndarray, *, mode: str = "auto", photographic: bool = False) -> EnhanceResult:
    """Upscale and restore ``rgba`` when it needs it. Never raises: on any
    failure the original image is returned and the report says why."""
    started = time.perf_counter()
    blur = measure_blur(rgba)
    report: dict = {"mode": mode, **blur}

    def skipped(reason: str) -> EnhanceResult:
        report.update(applied=False, reason=reason)
        return EnhanceResult(image=rgba, applied=False, report=report)

    if mode == "never":
        return skipped("disabled for this request")
    if not settings.enhance:
        return skipped("disabled on this server (VEC_ENHANCE=false)")
    if mode != "always":
        wanted, reason = _should_enhance(blur)
        if not wanted:
            return skipped(reason)
        report["reason"] = reason

    model = PHOTO_MODEL if photographic else FLAT_MODEL
    try:
        image, details = _upscale(rgba, model, blur["blur_factor"])
    except Exception as exc:  # a broken model must never cost the conversion
        log.warning("Enhancement failed, tracing the original: %s", exc)
        return skipped(f"enhancement failed ({exc.__class__.__name__})")
    if image is None:
        return skipped("super-resolution model not installed")

    report.update(
        applied=True,
        model=model,
        **details,
        output_size=[int(image.shape[1]), int(image.shape[0])],
        enhance_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    report.setdefault("reason", "forced for this request")
    log.info(
        "Enhanced %dx%d -> %dx%d with %s (%s) in %.0f ms",
        rgba.shape[1], rgba.shape[0], image.shape[1], image.shape[0],
        model, report["reason"], report["enhance_ms"],
    )
    return EnhanceResult(image=image, applied=True, report=report)


def _upscale(rgba: np.ndarray, model: str, blur_factor: float):
    net = _net(model)
    if net is None:
        return None, {}

    height, width = rgba.shape[:2]
    longest = max(width, height)
    # The true resolution, never above the model's input cap or the source.
    # Only when the blur is real: a sharp image that is merely small still
    # measures a little soft (its anti-aliasing), and reducing a 300px
    # sticker by that 1.3x erased its one-pixel line work.
    reduce = max(1.0, blur_factor * _REDUCE_BIAS) if blur_factor >= _MIN_BLUR_FACTOR else 1.0
    target = min(longest / reduce, _MAX_MODEL_INPUT, longest)
    scale = target / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))

    rgb = rgba[..., :3]
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else None
    has_transparency = alpha is not None and bool((alpha < 255).any())
    if has_transparency:
        # Transparent pixels carry arbitrary colour (often black). The model
        # would read that as content and draw a dark fringe round the art.
        rgb = _fill_transparent(rgb, alpha >= 128)

    small = rgb if size == (width, height) else cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    source = small.astype(np.float32) / 255
    with _lock:  # one cv2.dnn.Net is not safe to run from two threads at once
        upscaled = _run_tiled(net, source)
    upscaled = _clamp_ringing(source, _restore_colors(source, upscaled))

    out_h, out_w = upscaled.shape[:2]
    cap = settings.absolute_max_dimension
    if max(out_w, out_h) > cap:
        shrink = cap / max(out_w, out_h)
        out_w, out_h = max(1, round(out_w * shrink)), max(1, round(out_h * shrink))
        upscaled = cv2.resize(upscaled, (out_w, out_h), interpolation=cv2.INTER_AREA)

    result = np.empty((out_h, out_w, 4), np.uint8)
    result[..., :3] = np.clip(upscaled * 255 + 0.5, 0, 255).astype(np.uint8)
    result[..., 3] = (
        cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if has_transparency
        else 255
    )
    return result, {"model_input": [size[0], size[1]]}


def _run_tiled(net, rgb: np.ndarray) -> np.ndarray:
    """Run the 4x model over float RGB in [0, 1], tile by tile."""
    height, width = rgb.shape[:2]
    out = np.empty((height * MODEL_SCALE, width * MODEL_SCALE, 3), np.float32)
    for y0 in range(0, height, _TILE):
        for x0 in range(0, width, _TILE):
            y1, x1 = min(y0 + _TILE, height), min(x0 + _TILE, width)
            py0, px0 = max(0, y0 - _TILE_PAD), max(0, x0 - _TILE_PAD)
            py1, px1 = min(height, y1 + _TILE_PAD), min(width, x1 + _TILE_PAD)
            tile = rgb[py0:py1, px0:px1].transpose(2, 0, 1)[None]
            net.setInput(np.ascontiguousarray(tile))
            result = net.forward()[0].transpose(1, 2, 0)
            s = MODEL_SCALE
            out[y0 * s:y1 * s, x0 * s:x1 * s] = result[
                (y0 - py0) * s:(y1 - py0) * s, (x0 - px0) * s:(x1 - px0) * s
            ]
    return np.clip(out, 0.0, 1.0)


def _restore_colors(source: np.ndarray, upscaled: np.ndarray) -> np.ndarray:
    """Add back the low-frequency colour difference between source and output."""
    height, width = source.shape[:2]
    reduced = cv2.resize(upscaled, (width, height), interpolation=cv2.INTER_AREA)
    drift = cv2.GaussianBlur(source, (0, 0), _COLOR_SIGMA) - cv2.GaussianBlur(
        reduced, (0, 0), _COLOR_SIGMA
    )
    drift = cv2.resize(
        drift, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return np.clip(upscaled + drift, 0.0, 1.0)


def _clamp_ringing(source: np.ndarray, upscaled: np.ndarray) -> np.ndarray:
    """Keep every output pixel inside the colour range of its source neighbourhood.

    The flat model sharpens by overshooting: it draws a dark rim about one
    source pixel wide along every edge (measured on a red ring on cream: red
    dipping to 173 where neither side is below 204). Quantization then keeps
    the rim as a colour of its own, and Potrace traces it as slivers along
    every outline - 59 paths instead of 14 on a plain badge. Bounding each
    channel by the min/max of the 3x3 source neighbourhood removes the
    overshoot without blunting the edge.

    Except on thin lines. Blur lightens a one-pixel dark line, so its source
    neighbourhood never holds its real darkness, and the bound would hold the
    restored line at its blurred grey - which quantization then splits into
    dashes. Thin dark and light structures are found with a morphological
    top-hat (zero on a step edge between two flat regions, large on a line
    narrower than the kernel) and only the bound on their far side is lifted.
    """
    kernel = np.ones((3, 3), np.uint8)
    size = (upscaled.shape[1], upscaled.shape[0])
    low = cv2.erode(source, kernel)
    high = cv2.dilate(source, kernel)

    luma = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_LINE_KERNEL, _LINE_KERNEL))
    dark_line = cv2.morphologyEx(luma, cv2.MORPH_BLACKHAT, line_kernel) > _LINE_CONTRAST
    light_line = cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, line_kernel) > _LINE_CONTRAST
    low[cv2.dilate(dark_line.astype(np.uint8), kernel) > 0] = 0.0
    high[cv2.dilate(light_line.astype(np.uint8), kernel) > 0] = 1.0

    low = cv2.resize(low, size, interpolation=cv2.INTER_LINEAR)
    high = cv2.resize(high, size, interpolation=cv2.INTER_LINEAR)
    return np.clip(upscaled, low, high)


def _fill_transparent(rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Give transparent pixels the colour of the nearest artwork (push-pull)."""
    weight = opaque.astype(np.float32)
    color = rgb.astype(np.float32) * weight[..., None]
    pyramid = []
    while min(weight.shape) > 1:
        pyramid.append((color, weight))
        size = (max(1, weight.shape[1] // 2), max(1, weight.shape[0] // 2))
        color = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
        weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
    filled = color / np.maximum(weight, 1e-6)[..., None]
    for color, weight in reversed(pyramid):
        up = cv2.resize(filled, (weight.shape[1], weight.shape[0]), interpolation=cv2.INTER_LINEAR)
        filled = color + up * (1 - weight)[..., None]
    out = rgb.copy()
    out[~opaque] = np.clip(filled[~opaque] + 0.5, 0, 255).astype(np.uint8)
    return out
