"""Vectorization engines.

Importing this package registers every engine in the shared ``registry``.
"""

from .base import BaseVectorizer, EngineRegistry, EngineResult, registry

# Import for side effects: each module registers itself.
from . import vtracer_engine  # noqa: F401  (registers "vtracer")
from . import potrace_engine  # noqa: F401  (registers "potrace")
from . import photo_extractor  # noqa: F401  (registers "photo_extract")
from . import future_engines  # noqa: F401  (registers "starvector", "adobe_image_trace")

__all__ = [
    "BaseVectorizer",
    "EngineRegistry",
    "EngineResult",
    "registry",
]
