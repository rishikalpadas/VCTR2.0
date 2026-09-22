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
    # Exact colours quantization wrote into the image, as hex. None when
    # quantization is off, in which case there is no palette to hold the
    # tracer to.
    palette: list[str] | None = None


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
    palette: np.ndarray | None = None
    original_height, original_width = rgba.shape[:2]

    # Thin dark strokes are the first casualty of everything below: the
    # downscale averages them into their neighbours and the majority vote in
    # _smooth_labels then loses what is left. Record where they are at source
    # resolution now, while the information still exists.
    ink, dark = (
        _ink_mask(rgba, params.linework_max_luma, params.linework_min_contrast)
        if params.preserve_linework
        else (None, None)
    )

    working, base_scale = _resize(rgba, params, steps)
    working = _clamp_border(working)

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
        working, used_k, smoothed, palette = _quantize(
            working, params, analysis, supersample
        )
        steps.append(f"quantize(k={used_k})")
        if smoothed:
            steps.append(f"boundary_smooth(sigma={smoothed:.2g})")
        if ink is not None and palette is not None:
            working, restored = _restore_linework(
                working,
                ink,
                dark,
                palette,
                params.linework_max_luma,
                # Specified in output pixels like boundary_smooth_sigma, so it
                # has to move with the traced resolution the same way.
                sigma=params.linework_smooth_sigma * supersample,
                threshold=params.linework_coverage,
                # Already in tracing-resolution pixels, unlike the sigmas
                # above: _restore_linework converts it to source pixels itself
                # using the scale it can see.
                max_width=params.linework_max_width,
            )
            if restored:
                steps.append(f"linework_restored({restored:,}px)")

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
        palette=None if palette is None else [_to_hex(c) for c in palette],
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


def _clamp_border(rgba: np.ndarray) -> np.ndarray:
    """Overwrite the outermost pixel ring with its inward neighbour.

    A resampled or lossily-compressed image has a half-blended outermost row
    and column: the resampler had nothing beyond the canvas to average with, so
    the edge pixels end up a colour that appears nowhere else in the artwork.
    Quantization then either spends a cluster on that colour or snaps it to the
    nearest one, and either way the tracer wraps a one-pixel path around the
    whole canvas - hundreds of nodes describing an artifact of the crop.
    """
    if rgba.shape[0] < 3 or rgba.shape[1] < 3:
        return rgba
    result = rgba.copy()
    result[0] = result[1]
    result[-1] = result[-2]
    result[:, 0] = result[:, 1]
    result[:, -1] = result[:, -2]
    return result


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


def _assign(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-centre assignment for every sample, in memory-bounded chunks."""
    assigned = np.empty(samples.shape[0], dtype=np.int32)
    for start in range(0, samples.shape[0], _ASSIGN_CHUNK):
        chunk = samples[start : start + _ASSIGN_CHUNK]
        distances = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assigned[start : start + _ASSIGN_CHUNK] = distances.argmin(axis=1)
    return assigned


def _segment_offset(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Distance from ``point`` to the middle stretch of the ``start``-``end``
    segment, or infinity if it projects outside it. A colour sitting beyond an
    endpoint is darker or lighter than both parents, so it is not their blend.
    """
    span = end - start
    length_sq = float(span @ span)
    if length_sq <= 0:
        return float("inf")
    t = float((point - start) @ span) / length_sq
    if not 0.15 <= t <= 0.85:
        return float("inf")
    return float(np.linalg.norm(point - (start + t * span)))


# Share of a region that must survive erosion for it to count as a real area of
# artwork rather than a band hugging a boundary.
_BLEND_CORE_SHARE = 0.15


def _blend_centers(
    centers: np.ndarray,
    label_map: np.ndarray,
    *,
    max_share: float,
    max_offset: float,
    erode_px: int,
    dominance: float = 3.0,
) -> np.ndarray:
    """Indices of palette entries that are only the blend of two real colours.

    Anti-aliasing between two flat regions leaves a band of intermediate
    pixels, and k-means will spend a cluster on it: on dark-on-white artwork it
    lands a muddy grey between the ink and the paper. The band is a legitimate
    region - those pixels really are that colour - so neither
    _merge_close_centers (it sits far from both parents) nor boundary smoothing
    (which would only tidy its edges) removes it. It survives into the SVG as a
    halo hugging every stroke, in a colour the artwork never contained, and it
    costs a cluster the real colours needed.

    Colour geometry alone cannot identify one. Measured on a sticker whose
    trees are pale blue: the halo sat 3.5 off the line between its parents and
    the pale blue sat 8.9 off the line between *its* neighbours, so every
    threshold that caught the halo flattened the trees to white first. Being
    "between two other colours" is simply not rare.

    What does separate them is shape. A blend only ever exists along a
    boundary, so it is thin everywhere and erosion erases it; a real colour
    occupies area and keeps a core. Both tests have to pass: collinear *and*
    thin, with far fewer pixels than either parent.
    """
    count = centers.shape[0]
    if count < 3:
        return np.empty(0, dtype=np.int32)

    populations = np.bincount(
        label_map[label_map >= 0].ravel(), minlength=count
    ).astype(np.float64)
    total = populations.sum()
    if total <= 0:
        return np.empty(0, dtype=np.int32)

    points = centers.astype(np.float32)
    size = 2 * max(1, erode_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    doomed = []
    for index in range(count):
        if populations[index] / total > max_share:
            continue
        parents = [
            other
            for other in range(count)
            if other != index
            and populations[other] >= populations[index] * dominance
        ]
        collinear = any(
            _segment_offset(points[index], points[a], points[b]) <= max_offset
            for position, a in enumerate(parents)
            for b in parents[position + 1 :]
        )
        if not collinear:
            continue

        mask = (label_map == index).astype(np.uint8)
        core = int(cv2.erode(mask, kernel).sum())
        if core <= populations[index] * _BLEND_CORE_SHARE:
            doomed.append(index)

    return np.array(doomed, dtype=np.int32)


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




# Coverage at which a pixel counts as the solid spine of a stroke, kept even
# when smoothing would otherwise erase it. Below the obvious 0.75 because a
# hairline that lands under a pixel wide at tracing resolution peaks around
# 0.67 - measured on a 2px rule in a 2400px source traced at 2200 - and the
# whole point of the spine is that such a stroke still survives.
_INK_SPINE = 0.6


def _to_hex(color: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in color[:3]))


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma as float32, for any array shaped ``(..., 3)``."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return rgb.astype(np.float32) @ weights


def _ink_mask(
    rgba: np.ndarray, max_luma: int, min_contrast: float, radius: int = 0
) -> np.ndarray:
    """Locate the artwork's line work at source resolution.

    A single global luminance cut does not work here, and the reason is worth
    keeping: a stroke's edge sits at the luminance half way between the ink and
    whatever it is drawn against, and that midpoint is different for every
    neighbour. Measured on sticker artwork - ink at luma 45, white 253, brown
    107 - the true edge is at 149 against the white and at 76 against the
    brown. One threshold at 100 therefore reads the same stroke as too thin
    against the paper and too fat against the brown, which is the stroke width
    pulsing along its length.

    So the cut is taken locally instead. Dilating and eroding the luminance
    gives the lightest and darkest value within a stroke's reach of each pixel;
    a pixel is ink when it falls in the darker half of *that* range. Two gates
    keep it honest: the local minimum has to be dark enough to be ink at all
    (so a brown-to-salmon boundary is not mistaken for a stroke), and the local
    range has to be wide enough to be a real edge rather than noise in a flat
    region.
    """
    luma = _luma(rgba[..., :3])
    # The window has to reach across the stroke to the fills on either side,
    # and no further. Too wide and it finds some *other*, lighter colour
    # nearby, which drags the midpoint up until the fill next to the stroke
    # falls below it and gets swallowed - the stroke comes back too heavy.
    radius = radius or max(2, round(max(luma.shape) / 800))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    lightest = cv2.dilate(luma, kernel)
    darkest = cv2.erode(luma, kernel)

    stroke = (
        (darkest <= max_luma)
        & ((lightest - darkest) >= min_contrast)
        & (luma <= (lightest + darkest) * 0.5)
    )
    # A plain cut at the same level, kept alongside. The contrast gate above
    # blinds the stroke mask to the *inside* of a wide dark shape - the window
    # never reaches anything lighter, so only a fringe a couple of pixels deep
    # survives. That fringe is indistinguishable from a hairline by shape
    # alone, so the "is this thick enough to leave alone" test in
    # _restore_linework needs the whole dark region, not the fringe.
    dark = luma <= max_luma

    if rgba.shape[2] == 4:
        opaque = rgba[..., 3] >= 128
        stroke &= opaque
        dark &= opaque
    return stroke, dark


def _ink_coverage(
    ink: np.ndarray, size: tuple[int, int], sigma: float, threshold: float
) -> np.ndarray:
    """Resample the ink mask to ``size`` and re-threshold it into a clean mask.

    Two things have to happen here or the restored stroke is worse than no
    stroke at all.

    *Resampling.* INTER_AREA reports the fraction of the destination pixel a
    stroke covers, which is exactly what a hairline needs - but only when
    downscaling. OpenCV falls back to nearest-neighbour when asked to enlarge
    with it, so at the usual settings (1600px source, 1100px working size, 2x
    supersample = 2200px traced) the mask was being blown up 1.375x with no
    interpolation at all. Every stroke edge arrived at the tracer as a blocky
    1.4px staircase, which is the visible sawtooth on VTracer's output and the
    lumpy, width-varying stroke on Potrace's.

    *Smoothing.* Every other region boundary in the image has been through the
    majority vote in _smooth_labels by this point; the restored stroke has not,
    so it alone still carries pixel-level wobble. Blurring the coverage field
    and re-thresholding is the equivalent operation for a binary mask: it moves
    the edge to the sub-pixel position the coverage implies instead of snapping
    it to the pixel grid, and it cannot introduce a colour.

    The threshold sits below 0.5 on purpose. Blurring a stroke only a few
    pixels wide pulls its centre value down, so cutting at 0.5 would thin the
    thinnest strokes back out of existence - the exact failure being fixed.
    """
    height, width = ink.shape
    if (width, height) != size:
        interpolation = cv2.INTER_AREA if size[0] < width else cv2.INTER_LINEAR
        coverage = cv2.resize(
            ink.astype(np.float32), size, interpolation=interpolation
        )
    else:
        coverage = ink.astype(np.float32)

    if sigma <= 0:
        return coverage >= threshold

    kernel = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(coverage, (kernel, kernel), sigma)
    # Smoothing may refine a stroke's edge; it must never delete the stroke. A
    # rule thinner than one pixel at tracing resolution has its peak pulled
    # below any sensible threshold by the blur and simply disappears - which is
    # the original bug, reintroduced by its own fix. Pixels the resample says
    # are solidly covered are kept regardless. For a stroke wide enough to
    # smooth, this spine sits inside the smoothed edge and changes nothing.
    return (blurred >= threshold) | (coverage >= _INK_SPINE)


def _restore_linework(
    rgba: np.ndarray,
    ink: np.ndarray,
    dark: np.ndarray | None,
    palette: np.ndarray,
    max_luma: int,
    *,
    sigma: float,
    threshold: float,
    max_width: float,
) -> tuple[np.ndarray, int]:
    """Stamp line work the pipeline lost back onto the quantized image.

    Only pixels that were ink in the source *and* no longer carry any dark
    palette entry are touched, so solid dark regions keep whichever dark entry
    quantization gave them. The stamp always reuses an existing palette entry,
    so this can never introduce a colour - and therefore never an extra traced
    layer - that quantization did not already produce.

    Restricted to *thin* ink, and that restriction is load-bearing. A dark
    shape wide enough to survive the pipeline on its own has already had its
    boundary smoothed by the majority vote in _smooth_labels; stamping the raw
    source mask back over it throws that away and reinstates the very pixel
    wobble smoothing exists to remove. Measured on noisy concentric rings,
    restoring everything made boundary smoothing a no-op - identical segment
    counts with it on and off. A morphological opening removes exactly the
    structures narrower than the kernel while leaving wider ones alone, so
    subtracting it isolates the strokes that actually need rescuing.
    """
    height, width = rgba.shape[:2]

    if max_width > 0 and dark is not None:
        # The threshold is in traced pixels; the masks are at source resolution.
        scale = width / ink.shape[1]
        radius = max(1, round(max_width / (2 * scale)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        # Opening keeps only what is wider than the kernel. Dilating the result
        # back out covers the fringe of those wide shapes, which is the part
        # the stroke mask actually holds.
        thick = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        ink = ink & (cv2.dilate(thick, kernel) == 0)

    if ink.shape != (height, width) or sigma > 0:
        ink = _ink_coverage(ink, (width, height), sigma, threshold)

    luma = _luma(palette)
    dark = np.flatnonzero(luma <= max_luma)
    if dark.size == 0:
        # Nothing dark survived quantization. Stamping here would have to invent
        # a colour, which is exactly the extra-layer problem this avoids.
        return rgba, 0

    result = rgba.copy()
    rgb = result[..., :3]

    already_dark = np.zeros((height, width), dtype=bool)
    for index in dark:
        already_dark |= np.all(rgb == palette[index], axis=-1)

    lost = ink & ~already_dark
    if result.shape[2] == 4:
        lost &= result[..., 3] > 0
    if not lost.any():
        return rgba, 0

    rgb[lost] = palette[dark[int(np.argmin(luma[dark]))]]
    return result, int(lost.sum())


def _quantize(
    rgba: np.ndarray,
    params: PreprocessParams,
    analysis: ImageAnalysis,
    supersample: float = 1.0,
) -> tuple[np.ndarray, int, float, np.ndarray | None]:
    """k-means colour quantization over the visible pixels only.

    Transparent pixels are excluded from clustering so a large removed
    background cannot steal a cluster centre from the actual artwork.

    Returns ``(image, k, applied_sigma, palette)``, where ``palette`` is the
    exact uint8 colours written into the image.
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
        return result, 0, 0.0, None

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
    assigned = _assign(samples, centers)

    # --- collapse near-duplicate palette entries --------------------------
    # Do this before smoothing: the halo bands are legitimate regions, so
    # smoothing would just give them tidier edges instead of removing them.
    centers, assigned = _merge_close_centers(
        centers, assigned, params.min_color_separation
    )

    # --- drop anti-aliasing blends ----------------------------------------
    if params.drop_blend_colors:
        label_map = np.full(result.shape[:2], -1, dtype=np.int32)
        label_map[visible_mask] = assigned
        doomed = _blend_centers(
            centers,
            label_map,
            max_share=params.blend_max_share,
            max_offset=params.blend_max_offset,
            # A halo is as wide as the anti-aliasing ramp the tracer sees, so
            # it grows with the supersample factor and the test has to grow
            # with it.
            erode_px=max(1, round(1.5 * supersample)),
        )
        if doomed.size and centers.shape[0] - doomed.size >= 2:
            kept = np.setdiff1d(np.arange(centers.shape[0]), doomed)
            log.info(
                "Dropped %d blend colour(s): %s",
                doomed.size,
                ", ".join(_to_hex(centers[i].astype(np.uint8)) for i in doomed),
            )
            centers = centers[kept]
            # Re-assign from the original colours rather than remapping labels:
            # each pixel of the band then lands on whichever parent it was
            # actually nearer, which is what splitting an anti-aliased edge
            # means. Remapping wholesale would shift the whole band one way.
            assigned = _assign(samples, centers)

    effective_k = int(centers.shape[0])
    # Write through the uint8 palette rather than converting per assignment, so
    # the colours reported back are byte-identical to the ones in the image.
    palette = centers.astype(np.uint8)

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
        rgb[smoothed_visible] = palette[smoothed[smoothed_visible]]
        result[..., :3] = rgb
        return result, effective_k, applied_sigma, palette

    rgb[visible_mask] = palette[assigned]
    result[..., :3] = rgb
    return result, effective_k, 0.0, palette


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
