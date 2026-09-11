"""Pipeline tests. No pytest dependency - plain unittest.

Run from the python-engine directory:

    venv\\Scripts\\python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

# Make the service modules importable when running from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from analysis import ImageKind, analyze  # noqa: E402
from errors import (  # noqa: E402
    CorruptImageError,
    UnknownPresetError,
    UnsupportedFormatError,
)
from presets import get_preset  # noqa: E402
from svg_optimizer import validate_svg  # noqa: E402
from vectorizer import vectorize_bytes  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def donut(size: int = 400) -> Image.Image:
    """Red ring on a flat cream background - has a genuine hole."""
    image = Image.new("RGB", (size, size), (245, 240, 230))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, size - 40, size - 40), fill=(230, 60, 60))
    draw.ellipse((140, 140, size - 140, size - 140), fill=(245, 240, 230))
    return image


class TestValidation(unittest.TestCase):
    def test_rejects_empty_upload(self):
        with self.assertRaises(CorruptImageError):
            vectorize_bytes(b"")

    def test_rejects_non_image_bytes(self):
        with self.assertRaises(UnsupportedFormatError):
            vectorize_bytes(b"this is definitely not an image, it is text")

    def test_rejects_gif_even_though_pillow_could_read_it(self):
        image = Image.new("RGB", (64, 64), (10, 20, 30))
        buffer = io.BytesIO()
        image.save(buffer, format="GIF")
        with self.assertRaises(UnsupportedFormatError):
            vectorize_bytes(buffer.getvalue())

    def test_rejects_truncated_png(self):
        data = png_bytes(donut())
        with self.assertRaises(CorruptImageError):
            vectorize_bytes(data[: len(data) // 3])

    def test_rejects_unknown_preset(self):
        with self.assertRaises(UnknownPresetError):
            vectorize_bytes(png_bytes(donut()), preset_name="does_not_exist")


class TestOutputIsRealVector(unittest.TestCase):
    """The point of the whole exercise: actual geometry, not a wrapped raster."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = vectorize_bytes(png_bytes(donut()), preset_name="flat_art")

    def test_parses_as_xml(self):
        ET.fromstring(self.outcome.svg)

    def test_contains_path_geometry(self):
        self.assertGreater(self.outcome.meta["path_count"], 0)
        self.assertIn("<path", self.outcome.svg)

    def test_contains_no_embedded_raster(self):
        self.assertNotIn("<image", self.outcome.svg)
        self.assertNotIn("data:image", self.outcome.svg)

    def test_has_viewbox_and_original_dimensions(self):
        root = ET.fromstring(self.outcome.svg)
        self.assertIsNotNone(root.get("viewBox"))
        self.assertEqual(root.get("width"), "400")
        self.assertEqual(root.get("height"), "400")

    def test_reports_original_size_in_meta(self):
        self.assertEqual(self.outcome.meta["original_width"], 400)
        self.assertEqual(self.outcome.meta["original_height"], 400)


class TestHolesAndBackground(unittest.TestCase):
    def test_enclosed_removal_turns_a_hole_into_a_real_cutout(self):
        outcome = vectorize_bytes(
            png_bytes(donut()),
            preset_name="flat_art",
            overrides={"remove_enclosed_background": True},
        )
        root = ET.fromstring(outcome.svg)
        subpath_counts = [
            element.get("d", "").count("m") + element.get("d", "").count("M")
            for element in root.iter()
            if element.tag.endswith("path")
        ]
        # At least one path must carry a second, oppositely-wound subpath.
        self.assertTrue(
            any(count >= 2 for count in subpath_counts),
            f"expected a multi-subpath cutout, got {subpath_counts}",
        )

    def test_background_removal_can_be_disabled(self):
        outcome = vectorize_bytes(
            png_bytes(donut()),
            preset_name="flat_art",
            overrides={"background": "never"},
        )
        self.assertFalse(outcome.meta["background"]["applied"])

    def test_background_removal_runs_on_flat_canvas(self):
        outcome = vectorize_bytes(png_bytes(donut()), preset_name="flat_art")
        self.assertTrue(outcome.meta["background"]["applied"])


class TestCoordinatePrecision(unittest.TestCase):
    """Regression: the optimizer was destroying the geometry it optimized.

    scour's precision option counts SIGNIFICANT DIGITS, not decimal places. It
    was set to 2, so on a 1017px canvas every number in the file was rounded to
    two significant figures: the root width became "1e3" (1000), and every path
    control point was snapped to the integer pixel grid. That single setting
    accounted for roughly two thirds of the visible jaggedness.
    """

    @staticmethod
    def source_image() -> Image.Image:
        # 1017 is deliberate: it is the width that exposed the bug, and it
        # rounds to 1e3 at 2 significant digits.
        image = Image.new("RGB", (1017, 640), (245, 240, 230))
        draw = ImageDraw.Draw(image)
        draw.ellipse((60, 60, 560, 560), fill=(230, 60, 60))
        draw.polygon([(700, 80), (960, 300), (700, 560)], fill=(46, 128, 196))
        return image

    @classmethod
    def setUpClass(cls):
        cls.outcome = vectorize_bytes(
            png_bytes(cls.source_image()), preset_name="flat_art"
        )
        cls.root = ET.fromstring(cls.outcome.svg)

    def test_root_dimensions_are_not_rounded_away(self):
        self.assertEqual(self.root.get("width"), "1017")
        self.assertEqual(self.root.get("height"), "640")

    def test_viewbox_numbers_are_exact(self):
        viewbox = self.root.get("viewBox", "")
        self.assertNotIn("e", viewbox.lower(), f"viewBox lost precision: {viewbox}")
        parts = viewbox.split()
        self.assertEqual(len(parts), 4)
        width, height = float(parts[2]), float(parts[3])
        self.assertEqual(width, self.outcome.meta["processed_width"])
        self.assertEqual(height, self.outcome.meta["processed_height"])

    def test_viewbox_aspect_matches_the_source(self):
        parts = [float(p) for p in self.root.get("viewBox").split()]
        source_aspect = 1017 / 640
        self.assertAlmostEqual(parts[2] / parts[3], source_aspect, places=2)

    @staticmethod
    def _fractional_share(svg: str) -> float:
        root = ET.fromstring(svg)
        numbers = [
            number
            for element in root.iter()
            if element.tag.endswith("path")
            for number in re.findall(r"-?\d+\.?\d*", element.get("d", ""))
        ]
        if not numbers:
            return 0.0
        return sum("." in n for n in numbers) / len(numbers)

    def test_no_number_uses_scientific_notation(self):
        """`1e3` in a coordinate means significant digits were truncated.

        This is the signature of the bug: the value is not just imprecise, it
        is wrong by 17 pixels.
        """
        for element in self.root.iter():
            for name in ("d", "transform", "viewBox", "width", "height"):
                value = element.get(name)
                if value:
                    self.assertNotRegex(
                        value,
                        r"\d[eE][-+]?\d",
                        f"{name} lost precision to scientific notation: {value}",
                    )

    def test_default_precision_beats_the_old_setting(self):
        """Comparative, so it cannot silently pass if the defaults drift."""
        legacy = vectorize_bytes(
            png_bytes(self.source_image()),
            preset_name="flat_art",
            overrides={"significant_digits": 2},
        )
        self.assertGreater(
            self._fractional_share(self.outcome.svg),
            self._fractional_share(legacy.svg),
            "current output is no more precise than the old 2-significant-digit "
            "setting",
        )


class TestBackgroundSafety(unittest.TestCase):
    """Regression: the flood fill ate artwork that resembled the background."""

    @staticmethod
    def dark_on_dark() -> Image.Image:
        """Near-black artwork on a near-black field, 15 levels apart."""
        image = Image.new("RGB", (600, 400), (43, 41, 39))
        draw = ImageDraw.Draw(image)
        draw.rectangle((150, 100, 450, 300), fill=(28, 27, 24))
        draw.ellipse((260, 160, 340, 240), fill=(250, 250, 248))
        return image

    def test_tolerance_is_tightened_when_artwork_resembles_the_background(self):
        outcome = vectorize_bytes(
            png_bytes(self.dark_on_dark()),
            preset_name="flat_art",
            overrides={"background": "always"},
        )
        background = outcome.meta["background"]
        self.assertLess(
            background["tolerance"],
            18,
            "tolerance should be reduced when a colour sits close to the field",
        )

    def test_dark_artwork_survives_background_removal(self):
        outcome = vectorize_bytes(
            png_bytes(self.dark_on_dark()),
            preset_name="flat_art",
            overrides={"background": "always"},
        )
        # The dark rectangle is 25% of the canvas. If the fill leaked into it,
        # removal would approach 100% instead of stopping around the field.
        self.assertLess(
            outcome.meta["background"]["removed_ratio"],
            0.80,
            "flood fill leaked through the anti-aliased edge into the artwork",
        )

    def test_analysis_reports_the_colour_margin(self):
        array = np.array(self.dark_on_dark().convert("RGBA"))
        result = analyze(array)
        self.assertLess(result.border_color_margin, 30)


class TestSupersampling(unittest.TestCase):
    def test_supersample_enlarges_the_traced_canvas(self):
        outcome = vectorize_bytes(png_bytes(donut(600)), preset_name="logo")
        self.assertGreater(outcome.meta["supersample"], 1.0)
        self.assertGreater(outcome.meta["processed_width"], 600)
        # ...but the document still presents itself at the source size.
        root = ET.fromstring(outcome.svg)
        self.assertEqual(root.get("width"), "600")

    def test_engine_thresholds_scale_with_the_factor(self):
        """Pixel-denominated thresholds must move with the supersample factor.

        Without this, tracing at 2x leaves filter_speckle suppressing a quarter
        of the area it should and doubles the node count for no extra fidelity.
        """
        base = get_preset("logo").engine_params
        outcome = vectorize_bytes(png_bytes(donut(600)), preset_name="logo")
        used = outcome.meta["engine_meta"]["params"]
        factor = outcome.meta["supersample"]

        self.assertEqual(
            used["filter_speckle"], round(base.filter_speckle * factor * factor)
        )
        self.assertGreater(used["length_threshold"], base.length_threshold)

    def test_supersample_can_be_disabled(self):
        outcome = vectorize_bytes(
            png_bytes(donut(600)),
            preset_name="logo",
            overrides={"supersample": 1.0},
        )
        self.assertEqual(outcome.meta["supersample"], 1.0)


class TestOverrides(unittest.TestCase):
    def test_quantization_can_actually_be_turned_off(self):
        """Regression: `None` was treated as "not supplied", so passing it did
        nothing and quantization ran anyway."""
        outcome = vectorize_bytes(
            png_bytes(donut()),
            preset_name="flat_art",
            overrides={"quantize_colors": None},
        )
        steps = " ".join(outcome.meta["preprocess_steps"])
        self.assertNotIn("quantize", steps)

    def test_quantization_accepts_an_explicit_k(self):
        outcome = vectorize_bytes(
            png_bytes(donut()),
            preset_name="flat_art",
            overrides={"quantize_colors": 5},
        )
        self.assertIn("quantize(k=5)", outcome.meta["preprocess_steps"])


class TestPresets(unittest.TestCase):
    def test_every_preset_produces_valid_geometry(self):
        data = png_bytes(donut())
        for preset in ("standard", "logo", "flat_art", "typography", "detailed"):
            with self.subTest(preset=preset):
                outcome = vectorize_bytes(data, preset_name=preset)
                self.assertGreater(validate_svg(outcome.svg), 0)
                self.assertEqual(outcome.meta["preset_used"], preset)

    def test_line_art_preset_does_not_flood_the_canvas(self):
        """Regression: binary tracing on a transparent canvas traced one big box."""
        image = Image.new("RGB", (400, 300), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((60, 60, 240, 240), outline=(0, 0, 0), width=6)
        draw.line((280, 60, 360, 240), fill=(0, 0, 0), width=6)

        outcome = vectorize_bytes(png_bytes(image), preset_name="line_art")
        root = ET.fromstring(outcome.svg)
        paths = [el for el in root.iter() if el.tag.endswith("path")]
        self.assertGreaterEqual(len(paths), 2)
        # A canvas-sized rectangle would be a very short 'd' covering everything.
        longest = max(len(p.get("d", "")) for p in paths)
        self.assertGreater(longest, 120, "paths look degenerate (canvas rectangle?)")

    def test_auto_preset_resolves_to_a_concrete_preset(self):
        outcome = vectorize_bytes(png_bytes(donut()), preset_name="auto")
        self.assertEqual(outcome.meta["preset_requested"], "auto")
        self.assertNotEqual(outcome.meta["preset_used"], "auto")


class TestAnalysis(unittest.TestCase):
    def test_detects_flat_background(self):
        array = np.array(donut().convert("RGBA"))
        result = analyze(array)
        self.assertTrue(result.background_is_flat)
        self.assertIn(
            result.kind, {ImageKind.FLAT_GRAPHIC, ImageKind.TYPOGRAPHY}
        )

    def test_detects_transparency(self):
        image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(image).rectangle((50, 50, 150, 150), fill=(10, 200, 90, 255))
        result = analyze(np.array(image))
        self.assertTrue(result.has_alpha)


class TestGeneratedSamples(unittest.TestCase):
    """Runs only if tests/make_samples.py has been executed."""

    def test_sample_files_vectorize(self):
        files = sorted(SAMPLES.glob("sample_*"))
        if not files:
            self.skipTest("run `python tests/make_samples.py` first")
        for path in files:
            with self.subTest(sample=path.name):
                outcome = vectorize_bytes(path.read_bytes(), preset_name="auto")
                self.assertGreater(outcome.meta["path_count"], 0)
                self.assertNotIn("data:image", outcome.svg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
