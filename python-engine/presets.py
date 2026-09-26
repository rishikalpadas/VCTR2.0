"""Vectorization presets.

A preset is a *complete* description of one run of the pipeline: how to
preprocess, which engine to use, and how to tune that engine. Adding a new
preset means appending one entry to ``PRESETS`` - no other file changes.

Why presets at all? Because the target artwork is not one kind of image:

  * typography with crisp outlines needs tight corner handling and must NOT be
    blurred, or the letterforms go soft;
  * flat vector-style graphics benefit from colour quantization, which
    collapses anti-aliasing gradients into a handful of clean regions and
    dramatically reduces path count;
  * line art / doodles are fundamentally 1-bit and should be traced in binary
    mode, not as thousands of near-black colour layers;
  * dense, textured illustrations need a finer speckle filter so detail
    survives, at the cost of a bigger file.

Applying one aggressive setting to all of them produces mush.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from errors import UnknownPresetError

BackgroundMode = Literal["auto", "always", "never"]


@dataclass(frozen=True)
class VTracerParams:
    """Mirrors the vtracer Python binding 1:1 (see ``vtracer.pyi``)."""

    colormode: Literal["color", "binary"] = "color"
    hierarchical: Literal["stacked", "cutout"] = "stacked"
    mode: Literal["spline", "polygon", "none"] = "spline"
    filter_speckle: int = 4
    color_precision: int = 6
    layer_difference: int = 16
    corner_threshold: int = 60
    length_threshold: float = 4.0
    max_iterations: int = 10
    splice_threshold: int = 45
    path_precision: int = 3

    def as_kwargs(self) -> dict:
        return {
            "colormode": self.colormode,
            "hierarchical": self.hierarchical,
            "mode": self.mode,
            "filter_speckle": self.filter_speckle,
            "color_precision": self.color_precision,
            "layer_difference": self.layer_difference,
            "corner_threshold": self.corner_threshold,
            "length_threshold": self.length_threshold,
            "max_iterations": self.max_iterations,
            "splice_threshold": self.splice_threshold,
            "path_precision": self.path_precision,
        }


@dataclass(frozen=True)
class PotraceParams:
    """Mirrors the potrace 1.16 CLI 1:1. Defaults match the CLI's own defaults.

    Potrace only traces one colour (black) per invocation. The engine builds
    one binary mask per surviving colour in the already-quantized image and
    runs the binary once per mask, so these thresholds are pixel-denominated
    exactly like VTracer's and need the same supersample rescaling - see
    ``scale_engine_params``.
    """

    turnpolicy: Literal[
        "black", "white", "left", "right", "minority", "majority", "random"
    ] = "minority"
    # Speckle filter (an *area*, in pixels): suppress regions up to this size.
    turdsize: int = 2
    # Corner threshold. 0 = every vertex is a sharp corner; above ~1.334 =
    # always fit a curve. Opposite direction from VTracer's corner_threshold
    # (which is 0-100 and higher = more curve-friendly) - do not port the
    # number across, tune it fresh.
    #
    # Potrace's own default. No single value suits flat art: low enough to
    # keep letter and band corners (0.6 was used before), it also breaks every
    # gentle arc into polygon facets - an S came back as an octagon. So trace
    # for the arcs here and let ``corner_snap`` restore the corners.
    alphamax: float = 1.0
    # Bezier curve-fitting tolerance.
    opttolerance: float = 0.2
    # Where two straight traced edges are joined by curves spanning less than
    # this many pixels (rescaled), replace the curves with the edges' meeting
    # point - see ``potrace_engine._sharpen_corners``. This is what gives an
    # A's counter, a full stop and the tips of a bar their sharp corners back.
    # 0 disables it.
    corner_snap: float = 3.0
    # How far, in pixels, each fill layer runs on underneath the line work
    # (see ``_tuck_fills_under_ink``). A fill that stops exactly at a stroke
    # gets its outline fitted on the stroke's edge, and its curve fit then
    # bulges out past the ink into the next region - colour leaking across the
    # line work - or falls short and shows background. Tucked under, the fill
    # outline sits mid-stroke where the ink on top hides it. Measured on the
    # sticker artwork this replaced a stroked ink underlay: RMSE 17.75 ->
    # 14.44, 64 -> 35 paths, 112 -> 67 KB, and strokes back to their traced
    # weight instead of 2px heavier. 0 disables it. Rescaled for supersampling
    # like the stroke widths it has to reach across.
    ink_tuck: float = 3.0
    # Fill components smaller than this (an *area*, in pixels, rescaled like
    # turdsize) are merged into the nearest neighbouring fill before tracing.
    # Quantization strands a few pixels of the wrong colour at three-colour
    # junctions; turdsize would drop them and leave a hole instead.
    speck_area: int = 12
    # Trace thin line work by its centre and emit it as SVG strokes of one
    # width per connected network, instead of as filled outlines whose two
    # edges are fitted independently (see ``engines.centerline``). Thin means
    # narrower than ``centreline_max_width`` pixels (rescaled for
    # supersampling); anything wider - lettering, solid bars, a dark backdrop
    # - stays a filled shape.
    # How far, in pixels, each fill runs on under the layers painted after it
    # (see ``_underlap_later_layers``) so a seam between two fills with no
    # line work between them can never show background. Rescaled for
    # supersampling. Measured on the sticker artwork: unpainted pixels inside
    # the artwork 49 -> 0 (and 128 -> 0 with the background kept), with no change
    # in RMSE or file size. 0 disables it.
    fill_underlap: float = 1.5
    centreline: bool = True
    centreline_max_width: float = 3.0
    # Gaussian smoothing along the centreline, in pixels (rescaled).
    centreline_smooth: float = 1.0
    # How far, in pixels (rescaled), a fitted centreline curve may stray from
    # the skeleton. The skeleton of a raster stroke wobbles by about a pixel;
    # a curve held tighter than that reproduces the wobble as a wavy rule and
    # a lumpy circle, one held looser starts cutting real curvature.
    # Measured on sticker artwork: 0.8 -> 0.4 dropped RMSE 9.42 -> 8.25 with
    # rules still exact lines and the sun still an exact circle.
    centreline_tolerance: float = 0.4
    # Multiplier on the measured stroke width.
    stroke_weight: float = 1.0
    # Cap on distinct colour layers traced. Each layer is one subprocess call,
    # so this bounds worst-case latency on artwork that reached this engine
    # unquantized. Presets that already quantize (flat_art, logo, ...) never
    # get near it.
    max_layers: int = 32

    def as_cli_args(self) -> list[str]:
        return [
            "-z", self.turnpolicy,
            "-t", str(self.turdsize),
            "-a", str(self.alphamax),
            "-O", str(self.opttolerance),
        ]


@dataclass(frozen=True)
class PreprocessParams:
    # Working resolution of the artwork itself: the longest edge the image is
    # reduced to before tracing. Everything is mapped back via the viewBox.
    max_dimension: int = 1400

    # Supersampling factor applied on top of max_dimension.
    #
    # This is the single biggest lever on edge quality. A tracer sees a hard
    # pixel staircase on every diagonal and curve; fitting splines to that
    # staircase is what produces jagged letterforms and lumpy strokes. Upscaling
    # with interpolation first turns each staircase into a smooth ramp, so the
    # colour boundary lands at the sub-pixel position that best matches the
    # original anti-aliased edge. The result is then scaled back down by the
    # viewBox, halving the residual error in output units.
    #
    # It must be paired with scaled engine parameters - see
    # `scale_engine_params`. VTracer's thresholds are in pixels, so tracing at
    # 2x without scaling them just doubles the node count for no gain.
    supersample: float = 1.0

    # Small inputs (icons, tiny logos) trace poorly because anti-aliasing
    # dominates. Upscaling first gives the tracer more to work with.
    min_dimension_upscale: int = 512

    # Extra artifact suppression when the source was JPEG. White-on-dark
    # artwork is the worst case for JPEG: 8x8 DCT ringing puts a halo of
    # intermediate pixels around every letter, and the tracer faithfully
    # traces the halo.
    jpeg_cleanup: bool = True

    background: BackgroundMode = "auto"
    # Flood-fill colour tolerance, per channel, when removing a background.
    background_tolerance: int = 18

    # Extra pixels of background eaten past where the flood fill stopped.
    #
    # The boundary between artwork and canvas is an anti-aliasing ramp, not a
    # step. A tolerance tight enough to avoid leaking into the artwork stops
    # partway up that ramp, leaving a 1-2px band of half-blended pixels behind.
    # Alpha hardening then makes that band fully opaque, and the tracer renders
    # it as a separate, ragged, intermediate-coloured layer hugging every
    # shape - the most common cause of "the outline looks frayed".
    #
    # One pixel of dilation consumes the ramp. Raise it if you still see a
    # fringe; lower it to 0 to keep every last pixel of artwork edge.
    background_edge_bleed: int = 1

    # Also knock out *enclosed* regions that match the background colour -
    # the counters inside B, O, e, and gaps inside a wreath or frame.
    #
    # OFF by default and deliberately so. A flood fill from the border is
    # provably safe; this is not. An enclosed region matching the background
    # colour may be a letter counter (should go) or a genuine dark area of the
    # artwork (must stay) - a dark logo on a dark field is the classic trap.
    # Surfaced as a UI toggle instead of guessed at.
    remove_enclosed_background: bool = False
    # Only enclosed regions smaller than this share of the canvas are removed.
    # Letter counters are typically under 3%; a ring/donut hole lands around
    # 7-10%. Above that an enclosed same-colour region is far more likely to be
    # a real dark area of the artwork, so it is left alone.
    enclosed_max_area_ratio: float = 0.10

    # Edge-preserving denoise. 0 disables. Never use a plain Gaussian here -
    # it rounds off the corners of letterforms.
    bilateral_diameter: int = 0
    bilateral_sigma_color: int = 25
    bilateral_sigma_space: int = 25

    median_blur: int = 0  # odd kernel size, 0 = off

    # k-means colour quantization. None = off, "auto" = derive k from the
    # image's own colour statistics (see analysis.significant_colors).
    quantize_colors: int | Literal["auto"] | None = None
    quantize_min_k: int = 4
    quantize_max_k: int = 24

    # Minimum RGB distance between two surviving palette entries.
    #
    # k-means will happily spend clusters on colours the eye reads as one. On a
    # three-colour badge saved as JPEG it produced ten, and the extras landed
    # on the compression halo around every stroke: #CBCBFC alongside #CACAFA,
    # #C9C9F8, #DBDAF8. Each became a thin band hugging an outline, traced as
    # its own path. That is what reads as "bumps" along an otherwise clean
    # edge - not jaggedness in the curve, but a sliver of a slightly different
    # colour sitting on top of it.
    #
    # Smoothing cannot fix those; the band is a legitimate region, just a
    # pointless one. Merging centres closer than this distance removes them
    # outright. 0 disables. Raise it to flatten more aggressively; lower it if
    # genuinely close colours in the artwork are being fused.
    min_color_separation: float = 26.0

    # Drop palette entries that are only the blend of two other entries.
    #
    # min_color_separation above catches clusters that landed *next to* a real
    # colour. This catches the other failure: a cluster that landed exactly
    # between two of them, which is what an anti-aliased edge between two flat
    # regions looks like to k-means. On dark-on-white artwork that is a muddy
    # grey-brown, far enough from both parents to survive every other filter,
    # and it becomes a halo traced around every stroke in a colour the artwork
    # never had - while costing a cluster the real colours needed.
    #
    # Only safe on genuinely flat artwork. Leave off for anything with real
    # gradients or shading, where an intermediate tone is the point.
    drop_blend_colors: bool = False
    # A blend never covers much of the canvas; above this share of visible
    # pixels an intermediate colour is treated as real artwork.
    blend_max_share: float = 0.06
    # How far off the line between its two parents a colour may sit and still
    # count as their blend.
    blend_max_offset: float = 24.0

    # Majority-vote smoothing of the quantized colour regions, in OUTPUT
    # pixels (scaled internally by the supersample factor).
    #
    # This is what makes curves come out as curves. Quantization produces a
    # hard region boundary that still carries every wobble from the source
    # pixels - JPEG ringing, anti-aliasing, the odd stray pixel. A tracer reads
    # each wobble as a corner and joins them with straight segments, so a
    # perfectly circular ring comes back as a polygon: measured at 52% straight
    # line segments on a badge design made entirely of circles.
    #
    # Smoothing the region membership (not the colours) removes sub-pixel
    # boundary noise before the tracer ever sees it. On that badge it took line
    # segments from 52% to 3% and halved the file, at identical pixel fidelity.
    #
    # Only applies when quantization is on - it operates on the cluster labels.
    # Measured on a circular badge: raising it from 0.7 to 1.1 cut the node
    # count by 35% (2179 -> 1409 segments) and the file from 76 KB to 50 KB for
    # a ~2% RMSE cost, which is the trade you want on curved artwork. Past ~1.5
    # it starts rounding genuine detail and RMSE turns sharply worse.
    boundary_smooth_sigma: float = 0.0

    # Preserve thin dark line work through the pipeline.
    #
    # Keylines in flat artwork are 2-4px in the source and are the first thing
    # this pipeline destroys. The downscale to max_dimension averages them into
    # their neighbours, and boundary_smooth_sigma's majority vote then loses
    # what is left to the two large regions either side. The stroke does not
    # degrade gracefully - it either vanishes (a dark line between two mid-tone
    # fills snaps to one of them) or changes colour outright (a dark rule on
    # white snaps to whatever pale palette entry is nearest). A sticker coming
    # back with every keyline gone and the rule under its wordmark rendered in
    # pale blue is one bug, not two.
    #
    # When on, dark pixels are recorded at *source* resolution before any
    # resampling and stamped back after quantization, always reusing an
    # existing palette entry so no new colour - and so no new traced layer -
    # can appear. Only has an effect when quantization is on, since there is no
    # palette to stamp from otherwise.
    preserve_linework: bool = False
    # How dark the darkest pixel in a neighbourhood must be for that
    # neighbourhood to contain line work at all. Gates the *local minimum*, not
    # each pixel, so it only has to separate the ink from the darkest fill in
    # the artwork - 100 clears a mid-brown (~107) while catching black ink
    # (~45). The stroke's actual edge is found by local contrast; see _ink_mask.
    linework_max_luma: int = 100
    # Minimum luminance range in a neighbourhood before it is treated as an
    # edge rather than noise inside a flat region.
    linework_min_contrast: float = 40.0
    # Widest stroke, in TRACING-resolution pixels, that still counts as line
    # work needing rescue. Anything wider survives quantization on its own and
    # has already had its boundary smoothed; re-stamping it would undo that.
    #
    # Measured in traced pixels rather than output pixels (the convention the
    # sigmas above use) because that is the resolution the stroke has to
    # survive at, and _restore_linework can convert it to source pixels from
    # the scale it already knows.
    linework_max_width: float = 6.0
    # Smoothing of the restored stroke's edge, in OUTPUT pixels (scaled
    # internally by the supersample factor), matching boundary_smooth_sigma.
    #
    # Needed because the restored stroke is the one boundary in the image that
    # never went through the majority vote in _smooth_labels - it is stamped on
    # afterwards. Without this it reaches the tracer carrying raw pixel wobble,
    # and the tracer faithfully reproduces it: VTracer as a visible sawtooth
    # (measured: 40% of its segments came back as straight lines), Potrace as a
    # stroke whose width pulses along its length.
    #
    # Measured on sticker artwork, sigma 0 -> 0.4: Potrace 5329 -> 3422
    # segments (117 -> 80 KB) at identical RMSE, VTracer 3439 -> 2171 segments
    # with straight-line share falling 23% -> 7% and RMSE improving. Past 0.4 it
    # starts rounding the strokes themselves and both RMSE and segment count get
    # worse again, so this is a peak and not a "higher is smoother" dial.
    linework_smooth_sigma: float = 0.4
    # Share of a destination pixel a stroke must cover to survive. Below 0.5
    # because blurring a stroke a few pixels wide pulls its centre value down,
    # and cutting at the nominal half-coverage point would thin the thinnest
    # strokes back out of existence.
    linework_coverage: float = 0.4

    # Force the image to pure black/white before tracing (line art).
    binarize: bool = False
    binarize_block_size: int = 31
    binarize_c: int = 10

    # Semi-transparent pixels create halo paths around every shape. Anything
    # below this alpha becomes fully transparent, anything above fully opaque.
    alpha_threshold: int = 128
    harden_alpha: bool = True


@dataclass(frozen=True)
class OptimizeParams:
    # Drop paths whose bbox diagonal is < ratio * canvas diagonal.
    min_path_diagonal_ratio: float = 0.004

    # SIGNIFICANT digits kept by scour - NOT decimal places.
    #
    # This distinction is not cosmetic. scour's precision is significant
    # figures across the whole number, so on coordinates in the hundreds or
    # thousands a small value here destroys the geometry:
    #
    #   digits=2 -> "1017" becomes "1e3" (1000!), and a path coordinate of
    #               123.456 becomes 123 - every curve snapped to whole pixels
    #   digits=3 -> the root attributes survive, but coordinates are STILL
    #               integer-quantized
    #   digits=5 -> ~0.01px on a 1000px canvas, visually lossless
    #
    # Integer-snapped control points are exactly what produces jagged
    # letterforms and lumpy curves, so this floor matters more than any
    # tracer tuning. 5 is scour's own default; do not lower it to "save
    # bytes" without looking at the result at 400% zoom.
    significant_digits: int = 5

    collapse_groups: bool = True
    enabled: bool = True


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    engine: str = "vtracer"
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    engine_params: VTracerParams | PotraceParams = field(default_factory=VTracerParams)
    optimize: OptimizeParams = field(default_factory=OptimizeParams)


# ---------------------------------------------------------------------------
# The preset table
# ---------------------------------------------------------------------------

PRESETS: dict[str, Preset] = {
    "standard": Preset(
        name="standard",
        description=(
            "Balanced colour tracing. Safe default for most clean digital "
            "artwork: colourful graphics, stickers, mixed type + illustration."
        ),
        preprocess=PreprocessParams(
            max_dimension=1200,
            supersample=2.0,
            background="auto",
            bilateral_diameter=5,
            quantize_colors=None,
        ),
        engine_params=VTracerParams(
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=70,
            path_precision=3,
        ),
    ),
    "logo": Preset(
        name="logo",
        description=(
            "Badge and emblem artwork: flat colours, crisp lettering, thin "
            "rules and a small illustration. Supersamples heavily, keeps "
            "corners sharp and uses finer colour layers so small details "
            "survive."
        ),
        preprocess=PreprocessParams(
            max_dimension=1100,
            supersample=2.0,
            background="auto",
            # No bilateral: it is the JPEG cleanup stage's job, and blurring
            # unconditionally is what softens serifs.
            bilateral_diameter=0,
            jpeg_cleanup=True,
            quantize_colors="auto",
            quantize_max_k=10,
            boundary_smooth_sigma=1.1,
            preserve_linework=True,
            drop_blend_colors=True,
        ),
        engine_params=VTracerParams(
            filter_speckle=4,
            color_precision=8,
            # Finer than flat_art (24): the grey steps inside an illustration
            # are what get merged away at coarse layer differences.
            layer_difference=12,
            # High on purpose, and the opposite of the intuition that "lower =
            # sharper corners". A low threshold makes the tracer classify
            # boundary noise as corners and join them with straight segments;
            # measured, dropping from 70 to 40 cost 1.9 RMSE and turned half
            # the segments of a circular badge into polylines.
            corner_threshold=70,
            length_threshold=4.5,
            splice_threshold=45,
            path_precision=3,
        ),
        optimize=OptimizeParams(min_path_diagonal_ratio=0.003),
    ),
    "flat_art": Preset(
        name="flat_art",
        description=(
            "Flat-colour illustrations and stickers. Quantizes colours first, "
            "which collapses anti-aliasing bands into clean regions and cuts "
            "path count hard."
        ),
        preprocess=PreprocessParams(
            max_dimension=1200,
            supersample=2.0,
            background="auto",
            bilateral_diameter=5,
            quantize_colors="auto",
            quantize_max_k=12,
            boundary_smooth_sigma=1.0,
            preserve_linework=True,
            drop_blend_colors=True,
        ),
        engine_params=VTracerParams(
            filter_speckle=8,
            color_precision=8,
            layer_difference=24,
            corner_threshold=70,
            length_threshold=4.5,
            path_precision=3,
        ),
    ),
    "typography": Preset(
        name="typography",
        description=(
            "Lettering, logos and outlined type. Preserves sharp corners and "
            "counters (the holes in B, O, e); no blurring at all."
        ),
        preprocess=PreprocessParams(
            max_dimension=1100,
            supersample=2.0,
            min_dimension_upscale=700,
            background="auto",
            bilateral_diameter=0,  # never soften letterform edges
            quantize_colors="auto",
            quantize_max_k=10,
            # Lighter than logo/flat_art: serif brackets and spurs are only a
            # couple of pixels across. In practice the adaptive safety factor
            # usually zeroes this for real lettering anyway.
            boundary_smooth_sigma=0.8,
            preserve_linework=True,
            drop_blend_colors=True,
        ),
        engine_params=VTracerParams(
            filter_speckle=6,
            color_precision=8,
            layer_difference=22,
            # Not lower than the others: see the logo preset. Serif corners are
            # preserved by withholding smoothing (the adaptive safety factor
            # zeroes it for rectilinear type), not by hunting for corners.
            corner_threshold=60,
            length_threshold=4.0,
            splice_threshold=45,
            path_precision=3,
        ),
        optimize=OptimizeParams(min_path_diagonal_ratio=0.005),
    ),
    "line_art": Preset(
        name="line_art",
        description=(
            "Single-colour line work, doodles, outlines, icons. Binarized and "
            "traced in 1-bit mode - produces one clean black shape layer."
        ),
        preprocess=PreprocessParams(
            max_dimension=1400,
            supersample=1.5,
            background="never",  # binarization handles the background
            bilateral_diameter=0,
            binarize=True,
            quantize_colors=None,
        ),
        engine_params=VTracerParams(
            colormode="binary",
            mode="spline",
            filter_speckle=4,
            corner_threshold=45,
            length_threshold=4.0,
            path_precision=3,
        ),
    ),
    "detailed": Preset(
        name="detailed",
        description=(
            "Dense, textured or photographic-ish artwork. Keeps small detail "
            "at the cost of a much larger SVG and slower processing."
        ),
        preprocess=PreprocessParams(
            max_dimension=1600,
            supersample=1.5,
            background="never",
            bilateral_diameter=5,
            quantize_colors=None,
        ),
        engine_params=VTracerParams(
            filter_speckle=2,
            color_precision=8,
            layer_difference=8,
            corner_threshold=60,
            length_threshold=3.5,
            path_precision=4,
        ),
        optimize=OptimizeParams(
            min_path_diagonal_ratio=0.002,
            significant_digits=6,  # dense artwork: keep every bit of curve
        ),
    ),
}

# "auto" is resolved at request time by analysis.choose_preset(); it is listed
# here so the frontend can offer it and /presets can describe it.
AUTO_PRESET_NAME = "auto"

AUTO_PRESET_DESCRIPTION = (
    "Inspect the image and pick one of the presets below automatically "
    "(heuristic, not a trained classifier)."
)


def scale_engine_params(
    params: VTracerParams | PotraceParams, factor: float
) -> VTracerParams | PotraceParams:
    """Rescale pixel-denominated engine thresholds for a supersampled trace.

    Every one of these thresholds is expressed in pixels of the image actually
    handed to the tracer. Trace a 2x image without touching them and:

      * ``filter_speckle``/``turdsize`` (an *area*) suppresses a quarter of
        what it should, so noise that used to be filtered now survives as
        paths;
      * ``length_threshold`` (a *length*) stops merging short segments, so the
        node count roughly doubles for no extra fidelity.

    Net effect of forgetting this: a much larger file that is no smoother.
    Areas scale with the square of the factor, lengths linearly.
    """
    if factor <= 1.0:
        return params

    if isinstance(params, PotraceParams):
        return replace(
            params,
            turdsize=max(1, round(params.turdsize * factor * factor)),
            speck_area=round(params.speck_area * factor * factor),
            ink_tuck=params.ink_tuck * factor,
            centreline_max_width=params.centreline_max_width * factor,
            fill_underlap=params.fill_underlap * factor,
            centreline_smooth=params.centreline_smooth * factor,
            centreline_tolerance=params.centreline_tolerance * factor,
            corner_snap=params.corner_snap * factor,
        )

    return replace(
        params,
        filter_speckle=max(1, round(params.filter_speckle * factor * factor)),
        length_threshold=round(
            min(10.0, max(3.5, params.length_threshold * factor)), 2
        ),
    )


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise UnknownPresetError(
            f"Unknown preset '{name}'. "
            f"Valid presets: {AUTO_PRESET_NAME}, {', '.join(sorted(PRESETS))}."
        ) from None


def list_presets() -> list[dict]:
    items = [
        {
            "name": AUTO_PRESET_NAME,
            "description": AUTO_PRESET_DESCRIPTION,
            "engine": "auto",
        }
    ]
    items.extend(
        {"name": p.name, "description": p.description, "engine": p.engine}
        for p in PRESETS.values()
    )
    return items


# Overrides a caller may set per request. Deliberately narrow: a client cannot
# push the engine into a pathological configuration.
def _cast_quantize(value):
    """Accept "auto", an explicit k, or several spellings of "off"."""
    if value in (None, 0, "0", "off", "none", "null", False):
        return None
    if value == "auto":
        return "auto"
    return int(value)


_PRE_OVERRIDES = {
    "max_dimension": int,
    "supersample": lambda v: float(min(3.0, max(1.0, float(v)))),
    "background": str,
    "background_tolerance": int,
    "background_edge_bleed": int,
    "remove_enclosed_background": bool,
    "jpeg_cleanup": bool,
    "boundary_smooth_sigma": lambda v: float(min(3.0, max(0.0, float(v)))),
    "min_color_separation": lambda v: float(min(120.0, max(0.0, float(v)))),
    "quantize_colors": _cast_quantize,
    "preserve_linework": bool,
    "drop_blend_colors": bool,
    "linework_max_luma": lambda v: int(min(255, max(0, int(v)))),
    "linework_smooth_sigma": lambda v: float(min(3.0, max(0.0, float(v)))),
    "linework_coverage": lambda v: float(min(0.9, max(0.1, float(v)))),
}

# Keys where an explicit null is a meaningful value ("turn this off") rather
# than "not supplied". Without this, `{"quantize_colors": None}` was
# indistinguishable from omitting the key, so quantization could not be
# disabled through the API at all.
_NULLABLE_PRE = {"quantize_colors"}

_ENGINE_OVERRIDES = {
    "filter_speckle": int,
    "color_precision": int,
    "layer_difference": int,
    "corner_threshold": int,
    "length_threshold": float,
    "splice_threshold": int,
    "path_precision": int,
    "mode": str,
}

_POTRACE_ENGINE_OVERRIDES = {
    "turnpolicy": str,
    "turdsize": int,
    "alphamax": float,
    "opttolerance": float,
    "ink_tuck": lambda v: float(min(16.0, max(0.0, float(v)))),
    "speck_area": lambda v: int(min(1000, max(0, int(v)))),
    "fill_underlap": lambda v: float(min(8.0, max(0.0, float(v)))),
    "centreline": bool,
    "centreline_max_width": lambda v: float(min(20.0, max(1.0, float(v)))),
    "centreline_smooth": lambda v: float(min(5.0, max(0.0, float(v)))),
    "centreline_tolerance": lambda v: float(min(4.0, max(0.1, float(v)))),
    "corner_snap": lambda v: float(min(12.0, max(0.0, float(v)))),
    "stroke_weight": lambda v: float(min(3.0, max(0.2, float(v)))),
}

# A client may force a different tracing backend onto an existing preset (used
# by the engine-comparison test page to hold preprocessing identical while
# swapping only the tracer). Whitelisted for the same reason every other
# override is: the browser must not reach a backend we have not thought about.
_ALLOWED_ENGINES = {"vtracer", "potrace"}

# Per-image boundary smoothing for Potrace (see vectorizer.vectorize_bytes).
# The presets' 1.0-1.1 is tuned for VTracer. On a clean source Potrace does
# better with less - it fits its own curves and restores corners itself - but
# on a noisy one, less smoothing leaves the noise on every edge and Potrace
# traces it as a sawtooth. So both are traced and the light one is kept only
# if its SVG is at most this much larger. Measured on 13 test images: clean
# sources grew 0.99-1.08x, noisy ones 1.20-3.57x.
POTRACE_LIGHT_SMOOTH = 0.6
POTRACE_LIGHT_SMOOTH_MAX_GROWTH = 1.10

_ENGINE_DEFAULT_PARAMS: dict[str, type] = {
    "vtracer": VTracerParams,
    "potrace": PotraceParams,
}


def apply_overrides(preset: Preset, overrides: dict | None) -> Preset:
    """Return a copy of ``preset`` with per-request overrides applied."""
    if not overrides:
        return preset

    pre_changes = {
        key: cast(overrides[key])
        for key, cast in _PRE_OVERRIDES.items()
        if key in overrides
        and (overrides[key] is not None or key in _NULLABLE_PRE)
    }

    updated = preset
    if pre_changes:
        updated = replace(updated, preprocess=replace(updated.preprocess, **pre_changes))

    requested_engine = overrides.get("engine")
    if requested_engine in _ALLOWED_ENGINES and requested_engine != updated.engine:
        # Switching tracer families: the old engine_params dataclass does not
        # apply to the new one, so start from that engine's own defaults
        # rather than trying to carry fields across.
        updated = replace(
            updated,
            engine=requested_engine,
            engine_params=_ENGINE_DEFAULT_PARAMS[requested_engine](),
        )

    engine_override_map = (
        _POTRACE_ENGINE_OVERRIDES if updated.engine == "potrace" else _ENGINE_OVERRIDES
    )
    engine_changes = {
        key: cast(overrides[key])
        for key, cast in engine_override_map.items()
        if overrides.get(key) is not None
    }
    if engine_changes:
        updated = replace(
            updated, engine_params=replace(updated.engine_params, **engine_changes)
        )
    if overrides.get("optimize") is False:
        updated = replace(updated, optimize=replace(updated.optimize, enabled=False))
    if overrides.get("significant_digits") is not None:
        updated = replace(
            updated,
            optimize=replace(
                updated.optimize,
                significant_digits=int(overrides["significant_digits"]),
            ),
        )
    return updated
