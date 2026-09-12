"""Background removal for clean artwork.

Design decision worth stating explicitly: this is a **flood fill from the image
border**, not a global "delete every pixel matching the background colour".

That distinction is the whole point. Take the pink lettering sample: the canvas
is pink, and the counters inside the letters are also pink. A global colour
match would punch holes through the artwork. A flood fill only removes pink
that is *connected to the edge of the canvas*, so enclosed counters survive.

It is also conservative by default: in ``auto`` mode it refuses to act unless
the border is genuinely flat and the region it would remove is a plausible
background (neither a sliver nor almost the entire image).
"""

from __future__ import annotations

import cv2
import numpy as np

from analysis import ImageAnalysis
from logging_config import get_logger
from presets import PreprocessParams

log = get_logger(__name__)

# Guardrails for auto mode.
_MIN_REMOVED_RATIO = 0.04  # below this it was not really a background
_MAX_REMOVED_RATIO = 0.95  # above this we would be deleting the artwork
_MIN_BORDER_UNIFORMITY = 0.85


def remove_background(
    rgba: np.ndarray,
    params: PreprocessParams,
    analysis: ImageAnalysis,
) -> tuple[np.ndarray, dict]:
    """Return ``(rgba, report)``. The input is never mutated."""
    mode = params.background
    report: dict = {"mode": mode, "applied": False, "reason": None, "removed_ratio": 0.0}

    if mode == "never":
        report["reason"] = "disabled by preset or request"
        return rgba, report

    if analysis.has_alpha and analysis.transparent_ratio > 0.05:
        report["reason"] = "image already has a transparent background"
        return rgba, report

    if mode == "auto" and not analysis.background_is_flat:
        report["reason"] = (
            f"border is not flat enough "
            f"(uniformity {analysis.border_uniformity:.2f} < {_MIN_BORDER_UNIFORMITY})"
        )
        return rgba, report

    tolerance = _safe_tolerance(params.background_tolerance, analysis)
    report["tolerance"] = tolerance
    if tolerance < params.background_tolerance:
        report["tolerance_note"] = (
            f"tightened from {params.background_tolerance} to {tolerance}: the "
            f"artwork contains a colour only {analysis.border_color_margin} "
            f"levels from the background"
        )

    mask = _flood_fill_border_mask(rgba[..., :3], tolerance)
    removed_ratio = float(mask.mean())
    report["removed_ratio"] = round(removed_ratio, 4)

    if mode == "auto":
        if removed_ratio < _MIN_REMOVED_RATIO:
            report["reason"] = (
                f"flood fill only reached {removed_ratio:.1%} of the image - "
                f"no obvious background"
            )
            return rgba, report
        if removed_ratio > _MAX_REMOVED_RATIO:
            report["reason"] = (
                f"flood fill reached {removed_ratio:.1%} of the image - "
                f"refusing to erase the artwork"
            )
            return rgba, report

    enclosed_ratio = 0.0
    if params.remove_enclosed_background:
        enclosed = _enclosed_background_mask(
            rgba[..., :3],
            border_mask=mask,
            background_color=analysis.border_color,
            tolerance=params.background_tolerance,
            max_area_ratio=params.enclosed_max_area_ratio,
        )
        enclosed_ratio = float(enclosed.mean())
        mask = mask | enclosed
        removed_ratio = float(mask.mean())
        report["removed_ratio"] = round(removed_ratio, 4)

    report["enclosed_removed_ratio"] = round(enclosed_ratio, 4)

    # Eat the anti-aliasing ramp the flood fill could not cross. Without this
    # the leftover half-blended band becomes its own ragged traced layer.
    if params.background_edge_bleed > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(
            mask.astype(np.uint8), kernel, iterations=params.background_edge_bleed
        ).astype(bool)
        removed_ratio = float(mask.mean())
        report["removed_ratio"] = round(removed_ratio, 4)
        report["edge_bleed_px"] = params.background_edge_bleed

    result = rgba.copy()
    # Clear colour as well as alpha, so the tracer never sees a halo of the old
    # background colour bleeding out of anti-aliased edges.
    result[mask, 3] = 0
    report["applied"] = True
    report["reason"] = f"removed flat background ({removed_ratio:.1%} of canvas)"
    if enclosed_ratio > 0:
        report["reason"] += f", including enclosed areas ({enclosed_ratio:.1%})"
    log.info("Background removed: %.1f%% of canvas", removed_ratio * 100)
    return result, report


_MIN_TOLERANCE = 4


def _safe_tolerance(requested: int, analysis: ImageAnalysis) -> int:
    """Clamp the fill tolerance so it cannot cross into look-alike artwork.

    A flood fill spreads through any pixel within ``tolerance`` of the seed.
    When the artwork contains a colour close to the background - a dark
    illustration on a dark field is the textbook case - the anti-aliased
    boundary between them is a continuous ramp, and a tolerance wider than the
    gap lets the fill walk straight across it and erase the artwork.

    Half the measured margin keeps the fill on the background side of that
    ramp while still absorbing compression noise.
    """
    margin = analysis.border_color_margin
    if margin >= 255:
        return requested
    return int(max(_MIN_TOLERANCE, min(requested, margin // 2)))


def _enclosed_background_mask(
    rgb: np.ndarray,
    *,
    border_mask: np.ndarray,
    background_color: tuple[int, int, int],
    tolerance: int,
    max_area_ratio: float,
) -> np.ndarray:
    """Mask of small enclosed regions matching the background colour.

    These are the counters inside letterforms: pixels the same colour as the
    canvas, but walled off from the border so the flood fill never reached
    them. Only components below ``max_area_ratio`` of the canvas are taken -
    a large enclosed same-colour region is far more likely to be real artwork.
    """
    reference = np.array(background_color, dtype=np.int16)
    distance = np.abs(rgb.astype(np.int16) - reference).max(axis=2)
    candidates = (distance <= tolerance) & ~border_mask

    if not candidates.any():
        return np.zeros(rgb.shape[:2], dtype=bool)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), connectivity=8
    )

    total_pixels = rgb.shape[0] * rgb.shape[1]
    max_area = total_pixels * max_area_ratio

    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):  # 0 is the background of the labelling itself
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_area:
            keep[label] = True

    return keep[labels]


def _flood_fill_border_mask(rgb: np.ndarray, tolerance: int) -> np.ndarray:
    """Boolean mask of pixels reachable from the border within ``tolerance``.

    Seeds are spread along all four edges rather than just the corners, so a
    background that is interrupted at one corner still gets covered.
    """
    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # cv2.floodFill needs a mask 2px larger in each dimension.
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    lo = (tolerance,) * 3
    hi = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)

    step = max(1, min(height, width) // 12)
    seeds: list[tuple[int, int]] = []
    for x in range(0, width, step):
        seeds.append((x, 0))
        seeds.append((x, height - 1))
    for y in range(0, height, step):
        seeds.append((0, y))
        seeds.append((width - 1, y))

    working = bgr.copy()
    for x, y in seeds:
        # Skip seeds already swallowed by an earlier fill.
        if mask[y + 1, x + 1]:
            continue
        cv2.floodFill(working, mask, (int(x), int(y)), 0, lo, hi, flags)

    filled = mask[1:-1, 1:-1] > 0

    # Close 1px gaps left by anti-aliasing so the edge of the artwork is clean,
    # then erode back by the same amount to avoid eating into the artwork.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(
        filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    )
    return closed.astype(bool)
