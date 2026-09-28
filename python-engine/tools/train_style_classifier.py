"""Build models/style_prototypes.npz for style_classifier.py.

    python tools/train_style_classifier.py <folder of labelled images>

One reference direction per style: the mean CLIP embedding of that style's
labelled examples, plus half the mean embedding of a few text descriptions of
it. The text term steadies a style that has only a handful of examples.
Prints leave-one-out accuracy - each image classified by prototypes built
without it - so the number reported is an honest estimate, not a fit.

Labels are by filename. Images whose style is genuinely mixed (a flat poster
with a photo inset, glitter on flat lettering) are left out on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import open_clip  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from style_classifier import MODEL_NAME, PRETRAINED, PROTOTYPES_PATH  # noqa: E402

CLASSES = ["flat", "painted", "photo"]
LABELS = {
    "flat": [
        "friends.jpeg", "hello3.jpeg", "makediff.jpeg", "pant.jpeg", "PEST.jpeg",
        "sample (1).jpeg", "sample (12).jpg", "sample (18).jpeg", "sample (20).jpg",
        "sample (22).jpg", "sample (26).jpeg", "sample (28).jpeg", "sample (35).jpg",
        "sample (4).jpg", "sample (53).jpg", "sample (6).jpeg", "weakness.jpeg",
        "WhatsApp Image 2026-09-03 at 11.14.07 AM.jpeg",
    ],
    "painted": ["mimi.jpeg", "music.jpeg", "sample (24).jpeg", "sample (25).jpeg", "sample (30).jpeg"],
    "photo": ["focus.jpeg", "newyork.jpeg", "sample (10).jpeg", "sample (3).jpeg", "sample (4).jpeg"],
}
PROMPTS = {
    "flat": [
        "a flat vector graphic with solid colors",
        "a logo with flat colors and bold outlines",
        "a sticker design with solid color fills",
        "typography design with flat colors",
    ],
    "painted": [
        "a digital painting with soft shading and gradients",
        "an anime illustration with painted shading",
        "a watercolor illustration",
        "a detailed 3D rendered character illustration",
    ],
    "photo": [
        "a photograph",
        "a photo of a printed t-shirt",
        "a photo of fabric with embroidery",
        "a product photo of clothing",
    ],
}
TEXT_WEIGHT = 0.5


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main(folder: Path) -> None:
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    names, labels, vectors = [], [], []
    with torch.no_grad():
        for index, style in enumerate(CLASSES):
            for name in LABELS[style]:
                image = preprocess(Image.open(folder / name).convert("RGB")).unsqueeze(0)
                vectors.append(model.encode_image(image).numpy()[0])
                names.append(name)
                labels.append(index)
        text = np.stack([unit(unit(model.encode_text(tokenizer(PROMPTS[c])).numpy()).mean(0)) for c in CLASSES])
    vectors, labels = unit(np.array(vectors)), np.array(labels)

    def weights_without(skip: int | None) -> np.ndarray:
        protos = []
        for index in range(len(CLASSES)):
            members = (labels == index) & (np.arange(len(labels)) != skip)
            protos.append(unit(vectors[members].mean(0)))
        return np.stack(protos) + TEXT_WEIGHT * text

    wrong = []
    for i in range(len(labels)):
        predicted = int(np.argmax(weights_without(i) @ vectors[i]))
        if predicted != labels[i]:
            wrong.append(f"{names[i]}: {CLASSES[labels[i]]} -> {CLASSES[predicted]}")
    print(f"leave-one-out: {len(labels) - len(wrong)}/{len(labels)} correct")
    for line in wrong:
        print("  wrong:", line)

    PROTOTYPES_PATH.parent.mkdir(exist_ok=True)
    np.savez(PROTOTYPES_PATH, weights=weights_without(None).astype(np.float32), classes=np.array(CLASSES))
    print("saved", PROTOTYPES_PATH)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
