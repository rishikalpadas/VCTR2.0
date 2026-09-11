"""Placeholders for engines that will plug in later.

These exist to prove the abstraction holds. Each one is a complete
``BaseVectorizer`` that reports itself as unavailable and raises a typed 501 if
selected - no silent fallbacks, no invented APIs.

Replacing VTracer later is a two-line change in a preset
(``engine="starvector"``); nothing in Node, the frontend, or the pipeline moves.
"""

from __future__ import annotations

import numpy as np

from errors import EngineNotImplementedError
from presets import Preset

from .base import BaseVectorizer, EngineResult, registry


class StarVectorVectorizer(BaseVectorizer):
    """Transformer-based image-to-SVG model (StarVector and similar).

    Notes for whoever wires this up:
      * these models are GPU-hungry; keep them optional and never import torch
        at module import time, or the CPU-only deployment pays the cost;
      * they generate SVG *source code* directly, so the output still needs the
        same validation and cleanup pass the raster tracers get;
      * practical resolution limits are low (~512px), so tiling or a
        detail-preserving fallback is required for large artwork.
    """

    name = "starvector"
    description = (
        "PLANNED: neural image-to-SVG model. Requires a GPU and a large model "
        "download; deliberately not a dependency of this CPU-only POC."
    )

    def is_available(self) -> bool:
        return False

    def vectorize(self, rgba: np.ndarray, preset: Preset) -> EngineResult:
        raise EngineNotImplementedError(
            "The neural vectorization engine is not installed on this server."
        )


class AdobeImageTraceVectorizer(BaseVectorizer):
    """Adobe Illustrator / Firefly Image Trace via their hosted API.

    Notes: needs credentials in the environment, network egress, and per-call
    billing. Rate limits and latency make it a batch/background job rather than
    a synchronous request, so expect to move the Node route to a queue first.
    """

    name = "adobe_image_trace"
    description = (
        "PLANNED: hosted Adobe Image Trace. Requires API credentials and "
        "network access; billed per call."
    )

    def is_available(self) -> bool:
        return False

    def vectorize(self, rgba: np.ndarray, preset: Preset) -> EngineResult:
        raise EngineNotImplementedError(
            "The hosted Adobe vectorization engine is not configured."
        )


registry.register(StarVectorVectorizer())
registry.register(AdobeImageTraceVectorizer())
