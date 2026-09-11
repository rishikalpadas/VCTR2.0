"""VTracer engine.

VTracer (visioncortex) is a CPU-only Rust tracer with Python bindings. It is a
good default for this workload because it does colour *layer* tracing rather
than single-colour tracing, and it fits curves (splines) instead of emitting
polygon staircases.

The exact binding signature was taken from the installed package's own type
stub (``vtracer/vtracer.pyi``), not from memory:

    convert_raw_image_to_svg(img_bytes, img_format=None, colormode=None,
        hierarchical=None, mode=None, filter_speckle=None,
        color_precision=None, layer_difference=None, corner_threshold=None,
        length_threshold=None, max_iterations=None, splice_threshold=None,
        path_precision=None) -> str

We use the in-memory variant so nothing touches disk.

Known characteristics of the output, handled by ``svg_optimizer``:
  * the root element carries ``width``/``height`` but no ``viewBox``;
  * each path may carry its own ``transform="translate(x,y)"``;
  * holes are emitted as extra subpaths wound in the opposite direction, so
    the default ``nonzero`` fill rule renders counters and cutouts correctly.
"""

from __future__ import annotations

import time

import numpy as np

from errors import EngineError
from image_io import to_png_bytes
from logging_config import get_logger
from presets import Preset

from .base import BaseVectorizer, EngineResult, registry

log = get_logger(__name__)

try:
    import vtracer

    _VTRACER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - only on a broken install
    vtracer = None  # type: ignore[assignment]
    _VTRACER_IMPORT_ERROR = exc


class VTracerVectorizer(BaseVectorizer):
    name = "vtracer"
    description = (
        "VTracer 0.6.x - CPU colour-layer tracer with spline fitting. "
        "Strong on flat digital artwork, lettering and line work."
    )

    def is_available(self) -> bool:
        return vtracer is not None

    def vectorize(self, rgba: np.ndarray, preset: Preset) -> EngineResult:
        if vtracer is None:
            raise EngineError(
                "The vectorization engine is not available on this server.",
                detail=f"vtracer import failed: {_VTRACER_IMPORT_ERROR!r}",
            )

        params = preset.engine_params
        png_bytes = to_png_bytes(rgba)

        started = time.perf_counter()
        try:
            svg = vtracer.convert_raw_image_to_svg(
                png_bytes,
                img_format="png",
                **params.as_kwargs(),
            )
        except Exception as exc:
            log.exception("VTracer failed")
            raise EngineError(
                "Vectorization failed while tracing the image.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not svg or "<svg" not in svg:
            raise EngineError(
                "The vectorization engine returned no usable output.",
                detail=f"engine returned {len(svg) if svg else 0} bytes",
            )

        log.info(
            "VTracer traced %dx%d in %.0f ms (%d raw paths, %.1f KB)",
            rgba.shape[1],
            rgba.shape[0],
            elapsed_ms,
            svg.count("<path"),
            len(svg) / 1024,
        )

        return EngineResult(
            svg=svg,
            engine=self.name,
            meta={
                "engine_ms": round(elapsed_ms, 1),
                "raw_path_count": svg.count("<path"),
                "raw_svg_bytes": len(svg.encode("utf-8")),
                "params": params.as_kwargs(),
            },
        )


registry.register(VTracerVectorizer())
