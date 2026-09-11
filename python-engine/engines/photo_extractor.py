"""Photo / garment design extraction - INTENTIONALLY NOT IMPLEMENTED IN V1.

Why this is a stub and not a half-working feature
-------------------------------------------------
A photograph of a printed T-shirt is a fundamentally different problem from a
clean artwork PNG. The pixels you want are entangled with:

    * fabric weave texture modulating every colour,
    * folds and wrinkles warping the print non-rigidly,
    * a lighting gradient across the garment,
    * cast shadows and specular highlights,
    * perspective distortion from the camera angle,
    * the garment colour showing through screen-print halftones.

Running a raster tracer directly on that produces thousands of blobby paths
that follow the *lighting*, not the artwork. It is not a tuning problem; the
required information has to be recovered before tracing, not during.

Feeding a photo through the clean-artwork path still "works" in the sense that
it returns an SVG - the ``detailed`` preset does exactly that - but the result
is a posterized approximation of a photo, not extracted print-ready artwork.
The API labels it as such in ``meta.warnings`` rather than pretending
otherwise.

Planned pipeline for V2
-----------------------
    photo
      -> garment detection            (segmentation model, e.g. SAM / U2-Net)
      -> print/design localization    (saliency or text+graphic detection)
      -> perspective correction       (estimate the print plane, unwarp)
      -> fabric texture removal       (frequency-domain or guided filtering)
      -> lighting/shadow normalization (intrinsic image decomposition)
      -> artwork extraction           (alpha matting against garment colour)
      -> vectorization                (hand back to the existing engine layer)

Each of those is a separate, testable stage. They land here, behind the same
``BaseVectorizer`` interface the rest of the system already speaks, so nothing
upstream or downstream has to change when they arrive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from errors import EngineNotImplementedError
from presets import Preset

from .base import BaseVectorizer, EngineResult, registry


@dataclass
class ExtractionResult:
    """What a finished extractor will hand back to the vectorizer."""

    artwork_rgba: np.ndarray  # isolated print, transparent background
    confidence: float
    notes: list[str]


class PhotoDesignExtractor:
    """Recovers flat artwork from a photograph of a printed garment.

    Every stage below is a TODO. They are listed as real methods so the shape
    of the future implementation is visible, and so a contributor can fill them
    in one at a time without redesigning anything.
    """

    def extract(self, rgba: np.ndarray) -> ExtractionResult:
        raise NotImplementedError(
            "Photo design extraction is not implemented in V1."
        )

    # -- stages -------------------------------------------------------------

    def detect_garment(self, rgba: np.ndarray) -> np.ndarray:
        """TODO: segment the garment from the scene background.

        Candidate approach: a CPU-friendly U2-Net / ISNet salient-object model,
        or SAM with a centre-point prompt when a GPU is available.
        """
        raise NotImplementedError

    def localize_print(self, rgba: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
        """TODO: find the printed design region inside the garment mask.

        Candidate approach: colour-contrast saliency against the estimated
        garment base colour, plus a text detector (DBNet/CRAFT) so lettering is
        not clipped.
        """
        raise NotImplementedError

    def correct_perspective(self, rgba: np.ndarray, print_mask: np.ndarray) -> np.ndarray:
        """TODO: estimate the print plane and unwarp it to a frontal view.

        Folds make this non-planar, so a homography is only a first
        approximation; a thin-plate spline driven by the garment's shading
        gradient is the realistic target.
        """
        raise NotImplementedError

    def remove_fabric_texture(self, rgba: np.ndarray) -> np.ndarray:
        """TODO: suppress the periodic weave pattern.

        Candidate approach: notch-filter the periodic peaks in the FFT, or a
        guided filter using a texture-only estimate as guidance.
        """
        raise NotImplementedError

    def normalize_lighting(self, rgba: np.ndarray) -> np.ndarray:
        """TODO: flatten the illumination gradient and remove cast shadows.

        Candidate approach: intrinsic image decomposition (reflectance vs
        shading), keeping only the reflectance layer.
        """
        raise NotImplementedError

    def extract_artwork(self, rgba: np.ndarray, garment_color: np.ndarray) -> np.ndarray:
        """TODO: alpha-matte the design away from the garment base colour."""
        raise NotImplementedError


class PhotoExtractionVectorizer(BaseVectorizer):
    """Registered so the engine list is honest about what exists.

    Selecting it returns HTTP 501 with a clear message instead of silently
    falling back and producing something that looks like a failure of the
    clean-artwork pipeline.
    """

    name = "photo_extract"
    description = (
        "PLANNED (V2): extract printed artwork from a garment photograph "
        "before vectorizing. Not implemented - see photo_extractor.py."
    )

    def is_available(self) -> bool:
        return False

    def vectorize(self, rgba: np.ndarray, preset: Preset) -> EngineResult:
        raise EngineNotImplementedError(
            "Extracting artwork from a garment photograph is not supported yet. "
            "Upload the original artwork file, or use the 'detailed' preset to "
            "trace the photo as-is (approximate, not print-ready)."
        )


registry.register(PhotoExtractionVectorizer())
