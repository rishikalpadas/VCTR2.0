"""Reproduce the reported 'LENS HOOD' failure case.

Recreates the structural conditions of the user's sample:
  * dark near-uniform field (#2B2927) running to all four edges
  * white serif lettering (high contrast -> worst case for JPEG ringing)
  * thin olive strokes and arcs (1-3 px features that quantization mangles)
  * a greyscale illustration block with small internal detail
  * saved as JPEG at a compression ratio close to the original
    (1017x880 in ~66 KB)

Run:  python tests/make_logo_repro.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[2] / "samples"

WIDTH, HEIGHT = 1017, 880
DARK = (43, 41, 39)
WHITE = (252, 252, 250)
OLIVE = (138, 132, 89)
GREY_MID = (120, 118, 110)
GREY_DARK = (28, 27, 24)
GREY_LIGHT = (228, 228, 222)


def _serif(size: int):
    for candidate in ("georgiab.ttf", "timesbd.ttf", "georgia.ttf", "times.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), DARK)
    draw = ImageDraw.Draw(img)

    cx = WIDTH // 2

    # --- upper arc (thin olive stroke) -------------------------------------
    draw.arc((190, 120, 830, 760), start=185, end=355, fill=OLIVE, width=7)

    # --- camera body -------------------------------------------------------
    body = (330, 260, 690, 430)
    draw.rounded_rectangle(body, radius=14, fill=GREY_DARK)
    draw.rounded_rectangle((360, 215, 560, 275), radius=10, fill=GREY_LIGHT)
    draw.polygon([(455, 200), (530, 200), (560, 250), (430, 250)], fill=WHITE)

    # dials, viewfinder, small controls - the details reported as lost
    draw.ellipse((372, 196, 424, 232), fill=GREY_LIGHT)
    draw.ellipse((382, 204, 414, 224), fill=GREY_MID)
    draw.ellipse((596, 190, 668, 240), fill=GREY_LIGHT)
    draw.ellipse((610, 200, 654, 230), fill=GREY_DARK)
    draw.rounded_rectangle((580, 268, 646, 312), radius=6, fill=GREY_MID)
    draw.rounded_rectangle((592, 278, 634, 302), radius=4, fill=GREY_LIGHT)
    draw.polygon([(612, 330), (640, 330), (626, 300)], fill=GREY_LIGHT)
    draw.polygon([(392, 246), (420, 246), (406, 216)], fill=GREY_LIGHT)
    draw.rounded_rectangle((556, 372, 600, 392), radius=4, fill=OLIVE)
    draw.rounded_rectangle((650, 356, 672, 400), radius=4, fill=OLIVE)

    # lens barrel - concentric rings, the part reported as smeared
    draw.ellipse((360, 290, 520, 450), fill=GREY_LIGHT)
    draw.ellipse((372, 302, 508, 438), fill=GREY_DARK)
    draw.ellipse((386, 316, 494, 424), fill=(58, 56, 50))
    draw.ellipse((402, 332, 478, 408), fill=GREY_DARK)
    draw.ellipse((418, 348, 462, 392), fill=(70, 68, 60))

    # --- baseline rule -----------------------------------------------------
    draw.rectangle((150, 424, 870, 434), fill=OLIVE)

    # --- wordmark ----------------------------------------------------------
    font = _serif(150)
    text = "LENS HOOD"
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (bbox[2] - bbox[0]) // 2, 470), text, font=font, fill=WHITE)

    # --- lower rules + hood shape -----------------------------------------
    draw.rectangle((210, 660, 430, 670), fill=OLIVE)
    draw.rectangle((640, 660, 860, 670), fill=OLIVE)
    draw.arc((190, 340, 830, 830), start=25, end=155, fill=OLIVE, width=7)

    # petal hood outline: long smooth curves, the 'bumps' the user circled
    hood = [
        (250, 672), (300, 700), (340, 760), (360, 790), (395, 800),
        (430, 780), (450, 745), (470, 730), (560, 730), (580, 745),
        (600, 780), (635, 800), (670, 790), (690, 760), (730, 700),
        (780, 672),
    ]
    draw.line(hood, fill=OLIVE, width=9, joint="curve")
    draw.polygon(hood + [(780, 850), (250, 850)], fill=GREY_DARK)
    draw.line(hood, fill=OLIVE, width=9, joint="curve")

    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = build()

    png_path = OUT_DIR / "logo_repro.png"
    img.save(png_path)

    # Match the reported compression ratio: 1017x880 in roughly 66 KB.
    target = OUT_DIR / "logo_repro.jpg"
    for quality in (92, 88, 84, 80, 76, 72, 68, 64, 60):
        img.save(target, quality=quality, subsampling=2)
        size_kb = target.stat().st_size / 1024
        if size_kb <= 70:
            print(f"JPEG quality={quality} -> {size_kb:.1f} KB")
            break
    else:
        print(f"JPEG -> {target.stat().st_size / 1024:.1f} KB")

    print(f"PNG  (lossless reference) -> {png_path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
