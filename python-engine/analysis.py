"""Cheap statistical analysis of an input image.

Two jobs:

1. Produce numbers the rest of the pipeline needs anyway (does it already have
   an alpha channel? is there a flat border colour we could flood-fill away?
   how many distinct colours are actually in play?).

2. Guess what *kind* of artwork this is, so the ``auto`` preset can route it.

This is deliberately a heuristic, not a trained classifier. It runs in a few
milliseconds on a downsampled copy and is honest about its confidence. The
``ImageKind`` enum is the seam where a real model (a small CNN, a CLIP probe,
whatever) would slot in later without touching any other module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import cv2
import numpy as np

from logging_config import get_logger

log = get_logger(__name__)

# Analysis runs on a thumbnail: the statistics we need are scale-invariant and
# this keeps the whole thing under ~10 ms.
_ANALYSIS_MAX_DIM = 400

# Minimum share of visible pixels for a colour bucket to count as a real,
# intentional colour rather than a compression or anti-aliasing artifact.
_SIGNIFICANT_COLOR_SHARE = 0.004


class ImageKind(str, Enum):
    LINE_ART = "line_art"
    TYPOGRAPHY = "typography"
    LOGO = "logo"
    FLAT_GRAPHIC = "flat_graphic"
    DETAILED_ILLUSTRATION = "detailed_illustration"
    PRODUCT_PHOTO = "product_photo"
    # Not produced by the current heuristic - reserved for the future
    # photo-extraction pipeline (see engines/photo_extractor.py).
    T_SHIRT_PHOTO = "t_shirt_photo"
    FASHION_DESIGN_SHEET = "fashion_design_sheet"
    UNKNOWN = "unknown"


@dataclass
class ImageAnalysis:
    width: int
    height: int
    has_alpha: bool
    transparent_ratio: float
    unique_colors: int
    # Share of pixels belonging to the 8 most common colours. High = flat art.
    top_color_coverage: float
    # Colours that individually cover a meaningful share of the image. This is
    # the number that should drive quantization, NOT `unique_colors`: the
    # latter counts every JPEG ringing artifact and anti-aliasing step as a
    # distinct colour, which inflates k and hands the quantizer clusters to
    # spend on noise instead of artwork.
    significant_colors: int
    is_grayscale: bool
    colorfulness: float
    edge_density: float
    # Border statistics drive background removal.
    border_color: tuple[int, int, int]
    border_uniformity: float
    background_is_flat: bool
    # Chebyshev distance from the border colour to the nearest *other*
    # significant colour. Small = the artwork contains something nearly the
    # same shade as the background, so a flood fill must tread carefully.
    border_color_margin: int
    # Classification
    kind: ImageKind
    kind_confidence: float
    notes: list[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["border_color"] = list(self.border_color)
        return data


def _thumbnail(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[:2]
    scale = _ANALYSIS_MAX_DIM / max(height, width)
    if scale >= 1.0:
        return rgba
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(rgba, new_size, interpolation=cv2.INTER_AREA)


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk colourfulness metric (0 = grey, >40 = vivid)."""
    red, green, blue = (rgb[..., i].astype(np.float32) for i in range(3))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    return float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


def _border_stats(rgb: np.ndarray) -> tuple[tuple[int, int, int], float]:
    """Modal border colour and the share of border pixels close to it."""
    thickness = max(1, min(rgb.shape[0], rgb.shape[1]) // 50)
    border = np.concatenate(
        [
            rgb[:thickness, :, :].reshape(-1, 3),
            rgb[-thickness:, :, :].reshape(-1, 3),
            rgb[:, :thickness, :].reshape(-1, 3),
            rgb[:, -thickness:, :].reshape(-1, 3),
        ]
    )
    # Quantize to 16-level buckets so anti-aliasing noise does not split the mode.
    buckets = (border // 16).astype(np.int32)
    keys = buckets[:, 0] * 4096 + buckets[:, 1] * 64 + buckets[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    modal_key = values[counts.argmax()]
    modal_mask = keys == modal_key
    modal_color = border[modal_mask].mean(axis=0)

    distance = np.abs(border.astype(np.int16) - modal_color.astype(np.int16)).max(axis=1)
    uniformity = float((distance <= 20).mean())
    return tuple(int(c) for c in modal_color), uniformity


def _nearest_significant_distance(
    bucket_keys: np.ndarray,
    counts: np.ndarray,
    border_color: tuple[int, int, int],
    total: int,
) -> int:
    """How close the nearest significant non-background colour is.

    Drives the background flood-fill tolerance. If the artwork contains a
    colour only ~15 levels from the canvas colour - a near-black illustration
    on a near-black field, say - then a fill tolerance of 18 will walk straight
    through the anti-aliased boundary and erase the artwork. Knowing the margin
    lets the tolerance be tightened automatically instead of guessing.

    Returns 255 when nothing else is close (the safe, unconstrained case).
    """
    significant = counts / total >= _SIGNIFICANT_COLOR_SHARE
    if not significant.any():
        return 255

    keys = bucket_keys[significant]
    # Undo the 8-level bucket packing: key = r*1024 + g*32 + b, each //8.
    red = (keys // 1024) * 8
    green = ((keys % 1024) // 32) * 8
    blue = (keys % 32) * 8
    palette = np.stack([red, green, blue], axis=1).astype(np.int16)

    reference = np.array(border_color, dtype=np.int16)
    distances = np.abs(palette - reference).max(axis=1)
    # Ignore the background bucket itself (and its immediate neighbours).
    distances = distances[distances > 8]
    if distances.size == 0:
        return 255
    return int(distances.min())


def analyze(rgba: np.ndarray) -> ImageAnalysis:
    height, width = rgba.shape[:2]
    small = _thumbnail(rgba)
    rgb = small[..., :3]
    alpha = small[..., 3]

    transparent_ratio = float((alpha < 250).mean())
    has_alpha = transparent_ratio > 0.005

    # Only consider visible pixels when describing the artwork itself.
    visible = rgb[alpha > 128] if has_alpha else rgb.reshape(-1, 3)
    if visible.size == 0:
        visible = rgb.reshape(-1, 3)

    quantized = (visible // 8).astype(np.int32)
    keys = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]
    bucket_keys, counts = np.unique(keys, return_counts=True)
    unique_colors = int(counts.size)
    total = counts.sum()
    top_color_coverage = float(np.sort(counts)[::-1][:8].sum() / total)

    # A colour is "significant" if it covers at least this share of the visible
    # pixels on its own. Compression artifacts and anti-aliasing steps are
    # spread thinly across many buckets and fall below the line; real flat
    # regions sit far above it.
    significant_colors = int((counts / total >= _SIGNIFICANT_COLOR_SHARE).sum())

    channel_spread = np.abs(
        visible[:, 0].astype(np.int16) - visible[:, 1].astype(np.int16)
    ).mean() + np.abs(
        visible[:, 1].astype(np.int16) - visible[:, 2].astype(np.int16)
    ).mean()
    is_grayscale = bool(channel_spread < 6)

    colorfulness = _colorfulness(visible.reshape(-1, 1, 3))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float((edges > 0).mean())

    border_color, border_uniformity = _border_stats(rgb)
    background_is_flat = bool(border_uniformity >= 0.85)
    border_color_margin = _nearest_significant_distance(
        bucket_keys, counts, border_color, total
    )

    kind, confidence, notes = _classify(
        has_alpha=has_alpha,
        unique_colors=unique_colors,
        significant_colors=significant_colors,
        top_color_coverage=top_color_coverage,
        is_grayscale=is_grayscale,
        colorfulness=colorfulness,
        edge_density=edge_density,
        background_is_flat=background_is_flat,
        gray=gray,
        alpha=alpha,
    )

    analysis = ImageAnalysis(
        width=width,
        height=height,
        has_alpha=has_alpha,
        transparent_ratio=round(transparent_ratio, 4),
        unique_colors=unique_colors,
        top_color_coverage=round(top_color_coverage, 4),
        significant_colors=significant_colors,
        is_grayscale=is_grayscale,
        colorfulness=round(colorfulness, 2),
        edge_density=round(edge_density, 4),
        border_color=border_color,
        border_uniformity=round(border_uniformity, 4),
        background_is_flat=background_is_flat,
        border_color_margin=border_color_margin,
        kind=kind,
        kind_confidence=round(confidence, 2),
        notes=notes,
    )
    log.info(
        "Analysis: kind=%s conf=%.2f colors=%d (significant=%d) flat_bg=%s edges=%.3f",
        kind.value,
        confidence,
        unique_colors,
        significant_colors,
        background_is_flat,
        edge_density,
    )
    return analysis


def _classify(
    *,
    has_alpha: bool,
    unique_colors: int,
    significant_colors: int,
    top_color_coverage: float,
    is_grayscale: bool,
    colorfulness: float,
    edge_density: float,
    background_is_flat: bool,
    gray: np.ndarray,
    alpha: np.ndarray,
) -> tuple[ImageKind, float, list[str]]:
    """Rule-based routing. Ordered most-specific first."""
    notes: list[str] = []

    # --- line art: essentially two tones, thin strokes ---------------------
    histogram = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    histogram = histogram / max(histogram.sum(), 1)
    bimodality = float(np.sort(histogram)[::-1][:3].sum())

    if is_grayscale and bimodality > 0.88 and unique_colors < 48:
        notes.append("near-bilevel grayscale with thin strokes")
        return ImageKind.LINE_ART, 0.8, notes

    # --- product photo: many colours, soft gradients, no flat background ---
    if unique_colors > 3000 and not background_is_flat and not has_alpha:
        notes.append(
            "high colour count with no flat background - looks photographic; "
            "classical tracing will approximate, not extract, the artwork"
        )
        return ImageKind.PRODUCT_PHOTO, 0.6, notes

    # --- typography: flat palette, high edge density, lots of empty space ---
    empty_ratio = float((alpha < 128).mean()) if has_alpha else 1.0 - top_color_coverage
    if (
        unique_colors < 400
        and top_color_coverage > 0.82
        and edge_density > 0.03
        and empty_ratio > 0.25
    ):
        notes.append("few colours, dense edges, large empty areas - lettering-like")
        return ImageKind.TYPOGRAPHY, 0.65, notes

    # --- logo/emblem: flat palette but dense structure ---------------------
    # A badge is a flat graphic *and* a typography case at once: few real
    # colours, but lettering, thin rules and a small illustration all needing
    # sharp corners and fine colour layers. Routing it to plain flat_art
    # rounds the serifs and merges the illustration's greys together.
    if (
        significant_colors <= 12
        and top_color_coverage > 0.7
        and edge_density > 0.04
    ):
        notes.append(
            f"flat palette ({significant_colors} significant colours) with dense "
            f"edges - emblem/logo artwork"
        )
        return ImageKind.LOGO, 0.7, notes

    # --- flat graphic: small palette dominating the frame ------------------
    if unique_colors < 900 and top_color_coverage > 0.7:
        notes.append("small dominant palette - flat vector-style artwork")
        return ImageKind.FLAT_GRAPHIC, 0.7, notes

    # --- detailed illustration: everything else that is still artwork ------
    if colorfulness > 18 or unique_colors > 900:
        notes.append("rich palette or texture - treated as detailed illustration")
        return ImageKind.DETAILED_ILLUSTRATION, 0.55, notes

    notes.append("no strong signal; falling back to the balanced preset")
    return ImageKind.UNKNOWN, 0.3, notes


# Routing table: image kind -> preset name. This is the single place to change
# when a new preset or a real classifier arrives.
KIND_TO_PRESET: dict[ImageKind, str] = {
    ImageKind.LINE_ART: "line_art",
    ImageKind.TYPOGRAPHY: "typography",
    ImageKind.LOGO: "logo",
    ImageKind.FLAT_GRAPHIC: "flat_art",
    ImageKind.DETAILED_ILLUSTRATION: "detailed",
    ImageKind.PRODUCT_PHOTO: "detailed",
    ImageKind.T_SHIRT_PHOTO: "detailed",
    ImageKind.FASHION_DESIGN_SHEET: "detailed",
    ImageKind.UNKNOWN: "standard",
}


def choose_preset(analysis: ImageAnalysis) -> str:
    """Map an analysis result onto a preset name for the ``auto`` mode."""
    if analysis.kind_confidence < 0.5:
        return "standard"
    return KIND_TO_PRESET.get(analysis.kind, "standard")
