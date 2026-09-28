"""Optional learned artwork-style classifier.

The rule-based analysis cannot tell a painted illustration from flat artwork:
every statistic it has (colour counts, edge density, flat-region share,
quantization error) overlaps between an anime portrait with soft shading and
a keylined sticker, so painted art was routed to the ``logo`` preset and came
back as blotchy posterized patches.

This classifier looks at the image the way a person does. It embeds the
image with CLIP (LAION ViT-B-32, safetensors weights - loading never executes
code) and compares it with one reference direction per style - "flat",
"painted", "photo" - built by ``tools/train_style_classifier.py`` from
labelled examples plus text descriptions of each style, and stored in
``models/style_prototypes.npz``.

It is deliberately used for one decision only (see ``analysis.choose_preset``)
and only when confident, and it is optional: without the packages in
requirements-ml.txt, or with VEC_STYLE_CLASSIFIER=false, ``classify`` returns
None and routing is exactly the rule-based one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import settings
from logging_config import get_logger

log = get_logger(__name__)

PROTOTYPES_PATH = Path(__file__).resolve().parent / "models" / "style_prototypes.npz"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


@dataclass(frozen=True)
class StylePrediction:
    style: str
    # Score gap to the runner-up style. Measured on the labelled test set:
    # correct painted predictions 0.28-0.29, the closest wrong one 0.011.
    margin: float
    scores: dict[str, float]

    def to_dict(self) -> dict:
        return {"style": self.style, "margin": round(self.margin, 3), "scores": self.scores}


_lock = threading.Lock()
_state: dict = {}


def _load():
    """Load the model once per process. None if unavailable (cached)."""
    with _lock:
        if "model" in _state:
            return _state["model"]
        _state["model"] = None
        if not settings.style_classifier or not PROTOTYPES_PATH.exists():
            return None
        try:
            import open_clip
            import torch
        except ImportError:
            log.info("Style classifier off: packages in requirements-ml.txt are not installed")
            return None
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED
            )
            model.eval()
            data = np.load(PROTOTYPES_PATH)
            weights = data["weights"].astype(np.float32)
            classes = [str(c) for c in data["classes"]]
        except Exception as exc:  # network down on first download, corrupt cache...
            log.warning("Style classifier unavailable: %s", exc)
            return None
        torch.set_grad_enabled(False)
        _state["model"] = (model, preprocess, weights, classes, torch)
        log.info("Style classifier loaded (%s/%s)", MODEL_NAME, PRETRAINED)
        return _state["model"]


def embed(rgb: np.ndarray) -> np.ndarray | None:
    """Unit-length CLIP image embedding, or None if the classifier is off."""
    loaded = _load()
    if loaded is None:
        return None
    model, preprocess, _, _, torch = loaded
    from PIL import Image

    tensor = preprocess(Image.fromarray(np.ascontiguousarray(rgb))).unsqueeze(0)
    vector = model.encode_image(tensor).numpy()[0].astype(np.float32)
    return vector / np.linalg.norm(vector)


def classify(rgba: np.ndarray) -> StylePrediction | None:
    loaded = _load()
    if loaded is None:
        return None
    _, _, weights, classes, _ = loaded
    rgb = rgba[..., :3]
    if rgba.shape[2] == 4 and (rgba[..., 3] < 250).any():
        # Transparent pixels would read as black; judge the art on white.
        alpha = rgba[..., 3:4].astype(np.float32) / 255
        rgb = (rgb.astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    vector = embed(rgb)
    scores = weights @ vector
    order = np.argsort(-scores)
    return StylePrediction(
        style=classes[int(order[0])],
        margin=float(scores[order[0]] - scores[order[1]]),
        scores={c: round(float(s), 4) for c, s in zip(classes, scores)},
    )
