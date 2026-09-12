"""Reproduce the reported 'Be happy' badge case.

Matches the structural conditions of the reported sample:
  * 902x908, flat lilac field (204,204,253) running to all edges
  * two thin near-black concentric rings - long, smooth, high-curvature arcs
    where any boundary wobble is immediately visible
  * bold rounded lettering with a dark outline and white fill
  * a small leaf sprig (fine detail)
  * saved as JPEG at the reported compression ratio (~80 KB)

The rings are the important part: a circle has no corners at all, so every
straight segment or bump in the traced result is an artifact.

Run:  python tests/make_badge_repro.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[2] / "samples"

WIDTH, HEIGHT = 902, 908
LILAC = (204, 204, 253)
INK = (45, 42, 50)
WHITE = (252, 252, 250)

# Render oversized then downsample: gives clean anti-aliased edges, which is
# what a real vector-exported design looks like before JPEG gets to it.
SS = 4


def _rounded_font(size: int):
    for candidate in ("ariblk.ttf", "arialbd.ttf", "seguisb.ttf", "verdanab.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _outlined_text(draw, xy, text, font, outline_px):
    """White letters with a heavy dark outline, drawn by stroking."""
    draw.text(xy, text, font=font, fill=WHITE, stroke_width=outline_px,
              stroke_fill=INK, anchor="mm")


def _leaf_sprig(draw, cx, cy, scale):
    draw.line((cx - 150 * scale, cy, cx + 150 * scale, cy), fill=INK,
              width=int(5 * scale))
    for i in range(6):
        offset = (i + 1) * 24 * scale
        for direction in (-1, 1):
            x = cx + direction * offset
            draw.ellipse(
                (x - 26 * scale, cy - 34 * scale, x + 26 * scale, cy + 4 * scale),
                fill=INK,
            )
            draw.ellipse(
                (x - 26 * scale, cy - 4 * scale, x + 26 * scale, cy + 34 * scale),
                fill=INK,
            )


def build() -> Image.Image:
    width, height = WIDTH * SS, HEIGHT * SS
    img = Image.new("RGB", (width, height), LILAC)
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Thin border rule around the whole canvas.
    draw.rectangle((0, 0, width - 1, height - 1), outline=INK, width=2 * SS)

    # Two concentric rings, slightly offset like a hand-drawn badge.
    draw.ellipse((80 * SS, 90 * SS, 830 * SS, 840 * SS), outline=INK, width=4 * SS)
    draw.ellipse((96 * SS, 106 * SS, 812 * SS, 822 * SS), outline=INK, width=4 * SS)

    _outlined_text(draw, (cx, int(290 * SS)), "Be", _rounded_font(110 * SS), 9 * SS)
    _outlined_text(draw, (cx, int(455 * SS)), "hAppy", _rounded_font(130 * SS), 9 * SS)

    _leaf_sprig(draw, cx, int(625 * SS), SS)

    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = build()

    png_path = OUT_DIR / "badge_repro.png"
    img.save(png_path)

    target = OUT_DIR / "badge_repro.jpg"
    for quality in (95, 92, 90, 88, 85, 82, 78):
        img.save(target, quality=quality, subsampling=2)
        size_kb = target.stat().st_size / 1024
        if size_kb <= 85:
            print(f"JPEG quality={quality} -> {size_kb:.1f} KB")
            break
    else:
        print(f"JPEG -> {target.stat().st_size / 1024:.1f} KB")

    print(f"PNG (lossless reference) -> {png_path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
