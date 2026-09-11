"""Pipeline orchestration.

    bytes
      -> load + validate        (image_io)
      -> analyze                (analysis)
      -> resolve preset         (presets / analysis.choose_preset)
      -> preprocess             (preprocessing -> background)
      -> vectorize              (engines.registry)
      -> cleanup + validate     (svg_optimizer)
      -> VectorizeOutcome

This module owns *sequencing and reporting* only. Every piece of actual image
work lives in a dedicated module, and the engine is looked up by name, so the
orchestration does not change when either side is swapped out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from analysis import ImageKind, analyze, choose_preset
from engines import registry
from errors import EngineError, VectorizationError
from image_io import load_image
from logging_config import get_logger
from preprocessing import preprocess
from presets import AUTO_PRESET_NAME, apply_overrides, get_preset, scale_engine_params
from svg_optimizer import finalize_svg

log = get_logger(__name__)

# Kinds where classical raster tracing is an approximation, not an extraction.
_APPROXIMATION_KINDS = {
    ImageKind.PRODUCT_PHOTO,
    ImageKind.T_SHIRT_PHOTO,
    ImageKind.FASHION_DESIGN_SHEET,
}


@dataclass
class VectorizeOutcome:
    svg: str
    meta: dict = field(default_factory=dict)


def vectorize_bytes(
    data: bytes,
    *,
    preset_name: str = AUTO_PRESET_NAME,
    overrides: dict | None = None,
) -> VectorizeOutcome:
    started = time.perf_counter()
    warnings: list[str] = []

    # 1. Load + validate ----------------------------------------------------
    rgba, info = load_image(data)

    # 2. Analyze ------------------------------------------------------------
    analysis = analyze(rgba)

    # 3. Resolve the preset -------------------------------------------------
    requested = preset_name or AUTO_PRESET_NAME
    if requested == AUTO_PRESET_NAME:
        resolved_name = choose_preset(analysis)
        log.info(
            "Auto preset: kind=%s (conf %.2f) -> preset=%s",
            analysis.kind.value,
            analysis.kind_confidence,
            resolved_name,
        )
    else:
        resolved_name = requested

    preset = apply_overrides(get_preset(resolved_name), overrides)

    if analysis.kind in _APPROXIMATION_KINDS:
        warnings.append(
            "This looks like a photograph rather than clean digital artwork. "
            "Tracing will produce a posterized approximation, not extracted "
            "print-ready vector artwork - that requires the photo-extraction "
            "pipeline planned for V2."
        )

    # 4. Preprocess ---------------------------------------------------------
    prepared = preprocess(
        rgba, preset.preprocess, analysis, source_format=info["format"]
    )
    warnings.extend(prepared.warnings)

    # Supersampling changes the pixel scale the engine works in, so its
    # pixel-denominated thresholds have to move with it. Done here rather than
    # inside the engine so engines stay unaware of preprocessing.
    if prepared.supersample > 1.0:
        preset = replace(
            preset,
            engine_params=scale_engine_params(
                preset.engine_params, prepared.supersample
            ),
        )

    # 5. Vectorize ----------------------------------------------------------
    try:
        engine = registry.get(preset.engine)
    except KeyError as exc:
        raise EngineError(
            "The configured vectorization engine is not available.",
            detail=str(exc),
        ) from exc

    engine_result = engine.vectorize(prepared.image, preset)

    # 6. Cleanup + validate -------------------------------------------------
    optimized = finalize_svg(
        engine_result.svg,
        processed_width=prepared.processed_width,
        processed_height=prepared.processed_height,
        original_width=info["original_width"],
        original_height=info["original_height"],
        params=preset.optimize,
    )
    warnings.extend(optimized.warnings)

    if optimized.path_count > 12_000:
        warnings.append(
            f"The result contains {optimized.path_count:,} paths, which will be "
            f"slow to edit. Try the 'flat_art' preset or a smaller max "
            f"dimension for a simpler result."
        )

    total_ms = (time.perf_counter() - started) * 1000

    meta = {
        "preset_requested": requested,
        "preset_used": resolved_name,
        "engine": engine_result.engine,
        "source_format": info["format"],
        "original_width": info["original_width"],
        "original_height": info["original_height"],
        "processed_width": prepared.processed_width,
        "processed_height": prepared.processed_height,
        "scale": round(prepared.scale, 4),
        "supersample": round(prepared.supersample, 3),
        "path_count": optimized.path_count,
        "culled_paths": optimized.removed_paths,
        "svg_bytes": optimized.bytes_after,
        "svg_bytes_before_optimize": optimized.bytes_before,
        "size_reduction_pct": (
            round(
                100 * (1 - optimized.bytes_after / max(optimized.bytes_before, 1)), 1
            )
        ),
        "processing_ms": round(total_ms, 1),
        "preprocess_steps": prepared.steps,
        "background": prepared.background,
        "analysis": analysis.to_dict(),
        "engine_meta": engine_result.meta,
        "warnings": warnings,
    }

    log.info(
        "Done in %.0f ms: preset=%s paths=%d size=%.1f KB",
        total_ms,
        resolved_name,
        optimized.path_count,
        optimized.bytes_after / 1024,
    )
    return VectorizeOutcome(svg=optimized.svg, meta=meta)


def describe_engines() -> list[dict]:
    return registry.list()


__all__ = ["vectorize_bytes", "describe_engines", "VectorizeOutcome", "VectorizationError"]
