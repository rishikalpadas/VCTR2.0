"""Engine abstraction + registry.

The contract is intentionally tiny:

    RGBA numpy array + preset -> raw SVG string

Everything upstream (loading, analysis, preprocessing) and downstream (SVG
cleanup, validation, HTTP) is engine-agnostic. Swapping VTracer for an AI
vectorizer later means writing one class and registering it; the Node API and
the frontend do not change at all.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from presets import Preset


@dataclass
class EngineResult:
    svg: str
    engine: str
    # Free-form engine diagnostics surfaced in the API response.
    meta: dict = field(default_factory=dict)


class BaseVectorizer(abc.ABC):
    """Interface every vectorization engine implements."""

    #: Stable identifier used in presets and API responses.
    name: str = "base"

    #: Human-readable summary shown by GET /engines.
    description: str = ""

    @abc.abstractmethod
    def vectorize(self, rgba: np.ndarray, preset: Preset) -> EngineResult:
        """Convert an RGBA image into an SVG document string.

        Implementations must return real vector geometry. Embedding the raster
        (``<image>``, base64 data URIs) is rejected downstream by
        ``svg_optimizer.validate_svg``.
        """

    def is_available(self) -> bool:
        """Whether this engine can run in the current environment."""
        return True

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "available": self.is_available(),
        }


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, BaseVectorizer] = {}

    def register(self, engine: BaseVectorizer) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> BaseVectorizer:
        try:
            return self._engines[name]
        except KeyError:
            raise KeyError(
                f"Unknown engine '{name}'. Registered: {', '.join(sorted(self._engines))}"
            ) from None

    def list(self) -> list[dict]:
        return [engine.describe() for engine in self._engines.values()]


registry = EngineRegistry()
