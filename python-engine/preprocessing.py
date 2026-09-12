"""Image preprocessing.

Order matters here, and each step is opt-in via the preset rather than applied
blindly:

    resize -> background removal -> alpha hardening -> denoise
           -> colour quantization -> binarization

Rationale for the order:

* Resize first so every later step costs less and uses consistent kernel sizes.
* Background removal before denoise, because the flood fill wants the original
  hard edges.
* Alpha hardening before denoise, so semi-transparent fringe pixels are gone
  before any filter smears them back into the artwork.
* Quantization last (before binarization) because it should operate on already
  denoised colours - otherwise noise steals cluster centres.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from analysis import ImageAnalysis
from background import remove_background
from config import settings
from logging_config import get_logger
from presets import PreprocessParams

log = get_logger(__name__)


@dataclass
class PreprocessResult:
    image: np.ndarray  # RGBA uint8, this is what the engine traces
    processed_width: int
    processed_height: int
    scale: float  # processed / original
    # Supersampling factor actually applied (after the absolute size cap).
    # The engine's pixel-denominated thresholds must be scaled by this.
    supersample: float
    steps: list[str]
    background: dict
    warnings: list[str]


def preprocess(
    rgba: np.ndarray,
    params: PreprocessParams,
    analysis: ImageAnalysis,
    source_format: str = "png",
) -> PreprocessResult:
    """Prepare an image for tracing.

    Step order is deliberate:

      1. resize to the artwork working resolution
      2. JPEG artifact cleanup      - before anything reads the colours
      3. background removal         - wants the original hard edges
      4. alpha hardening            - kill fringe pixels before any filter
      5. edge-preserving denoise
      6. supersample                - AFTER denoise, so noise is not magnified
      7. colour quantization        - on clean, smooth-ramped pixels
      8. binarization
    """
    steps: list[str] = []
    warnings: list[str] = []
    original_height, original_width = rgba.shape[:2]

    working, base_scale = _resize(rgba, params, steps)

    if params.jpeg_cleanup and source_format in ("jpeg", "webp"):
        working = _jpeg_cleanup(working)
        steps.append("jpeg_cleanup")

    working, background_report = remove_background(working, params, analysis)
    if background_report["applied"]:
        steps.append(f"background_removed({background_report['removed_ratio']:.1%})")
    elif params.background != "never":
        warnings.append(f"Background kept: {background_report['reason']}.")

    if params.harden_alpha:
        working = _harden_alpha(working, params.alpha_threshold)
        steps.append("alpha_hardened")

    if params.bilateral_diameter > 0:
        working = _bilateral(working, params)
        steps.append(f"bilateral(d={params.bilateral_diameter})")

    if params.median_blur > 0:
        kernel = params.median_blur | 1  # force odd
        working[..., :3] = cv2.medianBlur(working[..., :3], kernel)
        steps.append(f"median_blur({kernel})")

    working, supersample = _supersample(working, params, steps)

    if params.quantize_colors is not None:
        working, used_k, smoothed = _quantize(working, params, analysis, supersample)
        steps.append(f"quantize(k={used_k})")
        if smoothed:
            steps.append(f"boundary_smooth(sigma={smoothed:.2g})")

    if params.binarize:
        working = _binarize(working, params)
        steps.append("binarize")

    height, width = working.shape[:2]
    log.info(
        "Preprocess: %dx%d -> %dx%d (supersample %.2gx) | %s",
        original_width,
        original_height,
        width,
        height,
        supersample,
        ", ".join(steps) or "none",
    )

    return PreprocessResult(
        image=working,
        processed_width=int(width),
        processed_height=int(height),
        scale=base_scale * supersample,
        supersample=supersample,
        steps=steps,
        background=background_report,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------


def _resize(
    rgba: np.ndarray, params: PreprocessParams, steps: list[str]
) -> tuple[np.ndarray, float]:
    """Reduce to the artwork working resolution (before any supersampling)."""
    height, width = rgba.shape[:2]
    longest = max(height, width)
    ceiling = min(params.max_dimension, settings.absolute_max_dimension)

    if longest > ceiling:
        scale = ceiling / longest
        # INTER_AREA is the correct choice for downscaling: it averages, which
        # suppresses aliasing instead of creating new speckles for the tracer.
        interpolation = cv2.INTER_AREA
    elif longest < params.min_dimension_upscale:
        scale = params.min_dimension_upscale / longest
        # CUBIC keeps edges crisper than LINEAR when enlarging small artwork.
        interpolation = cv2.INTER_CUBIC
    else:
        return rgba, 1.0

    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(rgba, new_size, interpolation=interpolation)
    steps.append(f"resize({width}x{height}->{new_size[0]}x{new_size[1]})")
    return resized, scale


def _supersample(
    rgba: np.ndarray, params: PreprocessParams, steps: list[str]
) -> tuple[np.ndarray, float]:
    """Upscale before tracing so edges land on sub-pixel boundaries.

    Returns the image and the factor actually applied, which may be lower than
    requested if the absolute size cap kicks in. The caller needs the real
    factor to rescale the engine's pixel thresholds.
    """
    requested = float(params.supersample)
    if requested <= 1.0:
        return rgba, 1.0

    height, width = rgba.shape[:2]
    longest = max(height, width)
    allowed = settings.absolute_max_dimension / longest
    factor = min(requested, allowed)

    if factor <= 1.01:
        steps.append(f"supersample_skipped(size_cap {settings.absolute_max_dimension}px)")
        return rgba, 1.0

    new_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    # CUBIC keeps edges crisper than LINEAR and does not ring like LANCZOS4,
    # which would reintroduce halos around exactly the high-contrast edges we
    # are trying to clean up.
    upscaled = cv2.resize(rgba, new_size, interpolation=cv2.INTER_CUBIC)
    steps.append(f"supersample({factor:.2g}x -> {new_size[0]}x{new_size[1]})")
    return upscaled, factor


def _jpeg_cleanup(rgba: np.ndarray) -> np.ndarray:
    """Suppress JPEG/WebP compression artifacts before tracing.

    Lossy codecs work on 8x8 blocks in the frequency domain. Around a hard
    edge - white lettering on a dark field is the pathological case - the
    truncated high frequencies come back as ringing: a halo of intermediate
    pixels and isolated speckles hugging every contour. A tracer cannot tell
    that halo from artwork, so it faithfully traces it, and you get wavy
    letter edges and lumps along thin strokes.

    Two cheap passes handle it:
      * a 3x3 median kills isolated mosquito speckles outright while leaving
        straight edges alone (unlike a mean/Gaussian blur, which smears them);
      * a bilateral pass with a wide colour sigma flattens the remaining
        ramp inside each flat region without crossing the real edge.
    """
    result = rgba.copy()
    rgb = result[..., :3]
    rgb = cv2.medianBlur(rgb, 3)
    rgb = cv2.bilateralFilter(rgb, d=7, sigmaColor=55, sigmaSpace=55)
    result[..., :3] = rgb
    return result


def _harden_alpha(rgba: np.ndarray, threshold: int) -> np.ndarray:
    """Make alpha strictly binary.

    Semi-transparent edge pixels otherwise become their own colour layers and
    the tracer wraps a thin ghost path around every shape.
    """
    result = rgba.copy()
    alpha = result[..., 3]
    opaque = alpha >= threshold
    result[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    # Neutralise the colour of fully transparent pixels so no colour bleeds in
    # from the removed background during filtering.
    result[~opaque, :3] = 0
    return result


def _bilateral(rgba: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Edge-preserving denoise.

    A bilateral filter smooths flat regions (killing JPEG mosquito noise and
    dithering that would otherwise explode into hundreds of tiny paths) while
    leaving hard colour boundaries alone.
    """
    result = rgba.copy()
    result[..., :3] = cv2.bilateralFilter(
        result[..., :3],
        d=params.bilateral_diameter,
        sigmaColor=params.bilateral_sigma_color,
        sigmaSpace=params.bilateral_sigma_space,
    )
    return result


def _auto_k(analysis: ImageAnalysis, params: PreprocessParams) -> int:
    """Pick a cluster count from the image's own colour statistics.

    Uses ``significant_colors`` (buckets that each cover a real share of the
    image), NOT ``unique_colors``. The old estimate was derived from the raw
    bucket count, which on a lossy source counts every ringing artifact as a
    colour: a six-colour logo saved as JPEG reported 437 buckets and got k=11.
    Those surplus clusters do not go to waste - they land on the artifact
    halos around high-contrast edges, turning each one into its own thin
    sliver path. That is where the lumpy outlines come from.

    A couple of headroom clusters are added so genuine shading inside an
    illustration is not crushed flat.
    """
    estimate = analysis.significant_colors + 2
    if analysis.is_grayscale:
        estimate = min(estimate, 8)
    return int(np.clip(estimate, params.quantize_min_k, params.quantize_max_k))


# Fitting k-means on every pixel of a supersampled image is pointless: cluster
# centres converge on a sample long before then. Fit on this many, assign all.
_KMEANS_FIT_SAMPLES = 150_000
_ASSIGN_CHUNK = 400_000


# Gaussian sigmas below this do nothing useful: the majority vote almost never
# flips, because a 3x3 kernel at sigma 0.5 still leaves ~60% of the weight on
# the centre pixel. Measured: sigma 0.5 produced byte-identical output to no
# smoothing at all. Anything under the bar is treated as "off" rather than
# silently costing a blur pass per cluster for no effect.
_MIN_EFFECTIVE_SIGMA = 0.8


def _merge_close_centers(
    centers: np.ndarray, labels: np.ndarray, min_separation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse palette entries the eye cannot tell apart.

    Greedy: repeatedly take the closest pair of surviving centres and, if they
    are nearer than ``min_separation``, fold the less populous one into the
    more populous one. The survivor keeps its own colour rather than a blend,
    so the dominant flat region is not shifted by absorbing its own edge halo.

    Returns the reduced palette and the labels remapped onto it.
    """
    if min_separation <= 0 or centers.shape[0] < 2:
        return centers, labels

    counts = np.bincount(labels, minlength=centers.shape[0]).astype(np.int64)
    # remap[i] is the surviving centre index that original centre i now uses.
    remap = np.arange(centers.shape[0])
    alive = np.ones(centers.shape[0], dtype=bool)

    while alive.sum() > 1:
        live_idx = np.flatnonzero(alive)
        live = centers[live_idx].astype(np.float32)
        diff = live[:, None, :] - live[None, :, :]
        distances = np.sqrt((diff**2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)

        flat = distances.argmin()
        a, b = np.unravel_index(flat, distances.shape)
        if distances[a, b] >= min_separation:
            break

        first, second = live_idx[a], live_idx[b]
        # Keep whichever covers more pixels; the other is the halo.
        keep, drop = (
            (first, second) if counts[first] >= counts[second] else (second, first)
        )
        remap[remap == drop] = keep
        counts[keep] += counts[drop]
        counts[drop] = 0
        alive[drop] = False

    survivors = np.flatnonzero(alive)
    compact = np.zeros(centers.shape[0], dtype=np.int32)
    compact[survivors] = np.arange(survivors.size)
    return centers[survivors], compact[remap][labels]


def _smooth_labels(
    labels: np.ndarray, cluster_count: int, sigma: float
) -> np.ndarray:
    """Majority-vote smoothing of a cluster-label map.

    Blur each cluster's binary membership mask and take the winner per pixel.
    Because the vote is over membership rather than colour, no new colours can
    appear - the output is still exactly the k cluster centres - while the
    boundary between regions loses its sub-pixel jitter.

    A plain blur of the *image* would not work: it would create intermediate
    colours along every boundary, which the tracer would then turn into extra
    sliver layers. Median-filtering the labels would be worse still, since
    label ids are nominal and their median is meaningless.
    """
    kernel = int(sigma * 6) | 1  # odd, ~3 sigma each side
    height, width = labels.shape
    best = np.full((height, width), -1.0, dtype=np.float32)
    winner = np.zeros((height, width), dtype=np.int32)

    for index in range(cluster_count):
        mask = (labels == index).astype(np.float32)
        blurred = cv2.GaussianBlur(mask, (kernel, kernel), sigma)
        improved = blurred > best
        best[improved] = blurred[improved]
        winner[improved] = index

    return winner


def _quantize(
    rgba: np.ndarray,
    params: PreprocessParams,
    analysis: ImageAnalysis,
    supersample: float = 1.0,
) -> tuple[np.ndarray, int, float]:
    """k-means colour quantization over the visible pixels only.

    Transparent pixels are excluded from clustering so a large removed
    background cannot steal a cluster centre from the actual artwork.

    Returns ``(image, k, applied_sigma)``.
    """
    k = (
        _auto_k(analysis, params)
        if params.quantize_colors == "auto"
        else int(params.quantize_colors)
    )
    k = max(2, k)

    result = rgba.copy()
    rgb = result[..., :3]
    visible_mask = result[..., 3] > 0

    samples = rgb[visible_mask].astype(np.float32)
    if samples.shape[0] < k:
        return result, 0, 0.0

    # --- fit centres on a subsample ---------------------------------------
    if samples.shape[0] > _KMEANS_FIT_SAMPLES:
        rng = np.random.default_rng(0)  # deterministic: same image, same output
        fit_idx = rng.choice(samples.shape[0], _KMEANS_FIT_SAMPLES, replace=False)
        fit_samples = np.ascontiguousarray(samples[fit_idx])
    else:
        fit_samples = samples

    # k-means++ seeding uses OpenCV's own global RNG, which is not seeded by
    # numpy. Without this the same image can produce different cluster centres
    # - and so a different SVG - on every run. Determinism matters both for
    # users re-exporting the same artwork and for the tests being meaningful.
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(
        fit_samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    # Stable ordering so cluster indices do not permute between runs.
    centers = centers[np.lexsort(centers.T[::-1])]
    centers = np.clip(centers, 0, 255).astype(np.float32)

    # --- assign every pixel to its nearest centre -------------------------
    assigned = np.empty(samples.shape[0], dtype=np.int32)
    for start in range(0, samples.shape[0], _ASSIGN_CHUNK):
        chunk = samples[start : start + _ASSIGN_CHUNK]
        distances = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assigned[start : start + _ASSIGN_CHUNK] = distances.argmin(axis=1)

    # --- collapse near-duplicate palette entries --------------------------
    # Do this before smoothing: the halo bands are legitimate regions, so
    # smoothing would just give them tidier edges instead of removing them.
    centers, assigned = _merge_close_centers(
        centers, assigned, params.min_color_separation
    )
    effective_k = int(centers.shape[0])

    # --- optional boundary smoothing --------------------------------------
    # Sigma is specified in output pixels, so scale it to the traced
    # resolution: at 2x supersampling a 0.7px request means 1.4px here.
    #
    # There used to be an additional reduction here, scaling smoothing down for
    # artwork with lots of axis-aligned edges. It was based on a measurement
    # (smoothing improved a curved emblem but wrecked a rectilinear one) that
    # two *later* fixes invalidated - correcting `corner_threshold` and adding
    # background edge bleed. Re-measured afterwards, smoothing improves both:
    # RMSE 8.15 -> 6.84 on the curved emblem and 9.42 -> 8.32 on the
    # rectilinear one. The gate was left in place from the stale numbers and
    # was cutting smoothing roughly in half on real badge artwork.
    applied_sigma = params.boundary_smooth_sigma * supersample
    if applied_sigma >= _MIN_EFFECTIVE_SIGMA:
        label_map = np.zeros(result.shape[:2], dtype=np.int32)
        label_map[visible_mask] = assigned
        # Transparent pixels get their own class so smoothing cannot drag the
        # artwork out over a removed background (or vice versa).
        if (~visible_mask).any():
            label_map[~visible_mask] = effective_k
            classes = effective_k + 1
        else:
            classes = effective_k

        smoothed = _smooth_labels(label_map, classes, applied_sigma)
        # Never let smoothing resurrect pixels that were made transparent.
        smoothed_visible = visible_mask & (smoothed < effective_k)
        rgb[smoothed_visible] = centers[smoothed[smoothed_visible]].astype(np.uint8)
        result[..., :3] = rgb
        return result, effective_k, applied_sigma

    rgb[visible_mask] = centers[assigned].astype(np.uint8)
    result[..., :3] = rgb
    return result, effective_k, 0.0


def _binarize(rgba: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Reduce to pure black ink on an opaque white canvas.

    Important: VTracer's ``colormode="binary"`` thresholds on *luminance* and
    ignores the alpha channel. Handing it black ink on a transparent canvas
    therefore reads as an all-black image and traces one canvas-sized
    rectangle. The output here must be black-on-white and fully opaque; the
    tracer emits only the dark shapes, so the resulting SVG still has a
    transparent background.

    Adaptive thresholding handles uneven lighting and scanned line art better
    than a single global cut-off. Otsu is the fallback when the adaptive result
    is degenerate (nearly all ink or no ink at all).
    """
    rgb = rgba[..., :3].copy()
    # Transparent pixels are background, not ink - force them to white before
    # thresholding, otherwise hardened alpha (RGB=0) would read as solid ink.
    if rgba.shape[2] == 4:
        rgb[rgba[..., 3] == 0] = 255

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    block = max(3, params.binarize_block_size | 1)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        params.binarize_c,
    )

    ink_ratio = float((adaptive > 0).mean())
    if ink_ratio < 0.002 or ink_ratio > 0.6:
        _, adaptive = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # Drop specks smaller than a few pixels before they become paths.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

    ink = adaptive > 0
    if rgba.shape[2] == 4:
        ink &= rgba[..., 3] > 0

    result = np.empty_like(rgba)
    result[..., :3] = np.where(ink[..., None], 0, 255).astype(np.uint8)
    result[..., 3] = 255  # opaque: binary tracing reads luminance, not alpha
    return result
