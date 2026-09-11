"""Generate synthetic test artwork into ../../samples/.

Deterministic stand-ins for the real upload categories, so the test suite has
fixtures without shipping binary assets:

    sample_typography.png  - outlined lettering with counters (holes)
    sample_flat_art.png    - flat colour shapes on a solid background
    sample_line_art.png    - thin black strokes on white
    sample_noisy.jpg       - JPEG-compressed, noisy, no flat background

Run:  python tests/make_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path(__file__).resolve().parents[2] / "samples"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def typography() -> Image.Image:
    """Lettering on a flat pink field - exercises counters + background removal."""
    img = Image.new("RGB", (900, 500), (247, 145, 145))
    draw = ImageDraw.Draw(img)
    font = _font(150)
    # "BOB" has three enclosed counters; they must survive the flood fill.
    draw.text((80, 60), "BOB", font=font, fill=(255, 255, 255))
    draw.text((80, 260), "GIRL", font=_font(120), fill=(60, 30, 40))
    draw.ellipse((700, 320, 840, 460), fill=(120, 40, 60))
    return img


def flat_art() -> Image.Image:
    img = Image.new("RGB", (800, 800), (245, 240, 230))
    draw = ImageDraw.Draw(img)
    draw.ellipse((120, 120, 480, 480), fill=(232, 70, 92))
    draw.ellipse((220, 220, 380, 380), fill=(245, 240, 230))  # hole
    draw.rectangle((500, 160, 700, 360), fill=(46, 128, 196))
    draw.polygon([(200, 560), (400, 560), (300, 720)], fill=(240, 190, 60))
    draw.ellipse((480, 520, 700, 740), fill=(70, 170, 120))
    return img


def line_art() -> Image.Image:
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 120, 420, 390), outline=(20, 20, 20), width=7)
    draw.arc((220, 200, 350, 330), start=20, end=160, fill=(20, 20, 20), width=6)
    draw.line((500, 150, 700, 420), fill=(20, 20, 20), width=6)
    draw.line((700, 150, 500, 420), fill=(20, 20, 20), width=6)
    for i in range(5):
        x = 140 + i * 120
        draw.line((x, 480, x + 70, 540), fill=(20, 20, 20), width=5)
    return img


def noisy() -> Image.Image:
    """Gradient + noise + blur: no flat background, photo-like statistics."""
    width, height = 700, 500
    xs = np.linspace(0, 255, width, dtype=np.float32)
    ys = np.linspace(0, 255, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    array = np.stack(
        [grid_x, grid_y, (grid_x + grid_y) / 2], axis=-1
    )
    rng = np.random.default_rng(1234)
    array += rng.normal(0, 18, array.shape)
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    draw = ImageDraw.Draw(img)
    draw.ellipse((250, 160, 450, 360), fill=(240, 60, 60))
    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    typography().save(OUT_DIR / "sample_typography.png")
    flat_art().save(OUT_DIR / "sample_flat_art.png")
    line_art().save(OUT_DIR / "sample_line_art.png")
    noisy().save(OUT_DIR / "sample_noisy.jpg", quality=72)
    for path in sorted(OUT_DIR.glob("sample_*")):
        print(f"wrote {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
