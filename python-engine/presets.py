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
    engine_params: VTracerParams = field(default_factory=VTracerParams)
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
            corner_threshold=60,
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
        ),
        engine_params=VTracerParams(
            filter_speckle=4,
            color_precision=8,
            # Finer than flat_art (24): the grey steps inside an illustration
            # are what get merged away at coarse layer differences.
            layer_difference=12,
            corner_threshold=40,
            length_threshold=4.0,
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
        ),
        engine_params=VTracerParams(
            filter_speckle=8,
            color_precision=8,
            layer_difference=24,
            corner_threshold=60,
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
        ),
        engine_params=VTracerParams(
            filter_speckle=6,
            color_precision=8,
            layer_difference=22,
            corner_threshold=38,  # sharper corner detection for serifs
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


def scale_engine_params(params: VTracerParams, factor: float) -> VTracerParams:
    """Rescale pixel-denominated engine thresholds for a supersampled trace.

    Every one of these thresholds is expressed in pixels of the image actually
    handed to the tracer. Trace a 2x image without touching them and:

      * ``filter_speckle`` (an *area*) suppresses a quarter of what it should,
        so noise that used to be filtered now survives as paths;
      * ``length_threshold`` (a *length*) stops merging short segments, so the
        node count roughly doubles for no extra fidelity.

    Net effect of forgetting this: a much larger file that is no smoother.
    Areas scale with the square of the factor, lengths linearly.
    """
    if factor <= 1.0:
        return params

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
    "remove_enclosed_background": bool,
    "jpeg_cleanup": bool,
    "quantize_colors": _cast_quantize,
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
    "path_precision": int,
    "mode": str,
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
    engine_changes = {
        key: cast(overrides[key])
        for key, cast in _ENGINE_OVERRIDES.items()
        if overrides.get(key) is not None
    }

    updated = preset
    if pre_changes:
        updated = replace(updated, preprocess=replace(updated.preprocess, **pre_changes))
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
