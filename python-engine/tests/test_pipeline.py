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

# Make the service modules importable when running from tests/, and this
# directory importable when unittest discovers with a different top level.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from analysis import ImageKind, analyze  # noqa: E402
from errors import (  # noqa: E402
    CorruptImageError,
    InvalidSvgError,
    UnknownPresetError,
    UnsupportedFormatError,
)
import pathstats  # noqa: E402
from presets import get_preset  # noqa: E402
from svg_export import (  # noqa: E402
    MAX_SVG_BYTES,
    _count_curve_operators,
    svg_to_pdf,
)
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


class TestCurveQuality(unittest.TestCase):
    """Regression: circles were coming back as polygons.

    A design made of rings was traced with 43-52% of its segments as straight
    lines. Two causes: no smoothing of the quantized region boundary, so every
    sub-pixel wobble read as a corner; and `corner_threshold=40`, which is far
    too eager to declare one.
    """

    @staticmethod
    def rings(size: int = 700) -> Image.Image:
        """Concentric rings: zero real corners anywhere in the artwork."""
        image = Image.new("RGB", (size, size), (204, 204, 253))
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 40, size - 40, size - 40), outline=(45, 42, 50), width=9)
        draw.ellipse((90, 90, size - 90, size - 90), outline=(45, 42, 50), width=9)
        draw.ellipse((210, 210, size - 210, size - 210), fill=(45, 42, 50))
        return image

    @classmethod
    def noisy_rings(cls) -> bytes:
        """The same rings as JPEG.

        Compression noise is the actual failure condition - on a lossless
        source there is no boundary jitter for smoothing to remove, so the
        regression only reproduces here.
        """
        buffer = io.BytesIO()
        cls.rings().save(buffer, format="JPEG", quality=82, subsampling=2)
        return buffer.getvalue()

    @staticmethod
    def line_share(svg: str) -> float:
        """Share of path segments that are straight lines.

        A proxy, not ground truth: scour legitimately rewrites a Bezier whose
        control points are collinear as a line, and that is visually identical.
        It is still the clearest signal available for "a circle came back as a
        polygon", which is why it is compared *relatively* below rather than
        against a fixed budget.
        """
        return pathstats.summarise(svg)["line_share"]

    def test_smoothing_reduces_polyline_output_on_noisy_curves(self):
        data = self.noisy_rings()
        smoothed = vectorize_bytes(data, preset_name="logo")
        raw = vectorize_bytes(
            data, preset_name="logo", overrides={"boundary_smooth_sigma": 0.0}
        )
        self.assertLess(
            self.line_share(smoothed.svg),
            self.line_share(raw.svg),
            f"smoothing did not reduce straight segments: "
            f"{self.line_share(smoothed.svg):.0%} vs "
            f"{self.line_share(raw.svg):.0%}",
        )

    def test_smoothing_reduces_segment_count_on_noisy_curves(self):
        """Fewer nodes for the same shape - the file gets simpler, not just smaller."""
        data = self.noisy_rings()
        smoothed = pathstats.summarise(
            vectorize_bytes(data, preset_name="logo").svg
        )
        raw = pathstats.summarise(
            vectorize_bytes(
                data, preset_name="logo", overrides={"boundary_smooth_sigma": 0.0}
            ).svg
        )
        self.assertLess(smoothed["segments"], raw["segments"])

    def test_smoothing_is_reported(self):
        outcome = vectorize_bytes(self.noisy_rings(), preset_name="logo")
        steps = " ".join(outcome.meta["preprocess_steps"])
        self.assertIn("boundary_smooth", steps)

    def test_smoothing_can_be_disabled(self):
        outcome = vectorize_bytes(
            self.noisy_rings(),
            preset_name="logo",
            overrides={"boundary_smooth_sigma": 0.0},
        )
        steps = " ".join(outcome.meta["preprocess_steps"])
        self.assertNotIn("boundary_smooth", steps)

    @staticmethod
    def _fill_count(svg: str) -> int:
        return len(set(re.findall(r'fill="(#[0-9A-Fa-f]{3,6})"', svg)))

    def test_palette_is_counted_from_flat_interiors(self):
        """Regression: the anti-aliasing ramp was being counted as colours.

        Every boundary is a ramp, and on a lossy source a wide one. Counting
        colours over all pixels made those ramp tones register as colours in
        their own right, so k-means spent most of its clusters describing the
        ramp - a three-colour badge produced dark, white, lilac and five
        intermediate greys, each traced as a thin band hugging an outline.
        """
        image = Image.new("RGB", (600, 600), (204, 204, 253))
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 40, 560, 560), outline=(45, 42, 50), width=9)
        draw.ellipse((90, 90, 510, 510), outline=(45, 42, 50), width=9)
        draw.ellipse((200, 200, 400, 400), fill=(252, 252, 250))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, subsampling=2)

        outcome = vectorize_bytes(buffer.getvalue(), preset_name="logo")
        # Three real colours; allow a little headroom, but nowhere near the
        # eight the whole-image count used to report.
        self.assertLessEqual(outcome.meta["analysis"]["significant_colors"], 5)

    def test_smoothing_does_not_expand_the_palette(self):
        """Smoothing votes on cluster membership, so it cannot invent colours.

        Note the output palette is still larger than k: VTracer's `stacked`
        hierarchy emits its own intermediate blend layers between colour
        regions regardless of how few colours the input had. What matters is
        that smoothing does not *add* to that - it should reduce it, by
        removing the noisy boundary pixels those layers were tracing.
        """
        data = self.noisy_rings()
        smoothed = vectorize_bytes(
            data, preset_name="logo", overrides={"quantize_colors": 4}
        )
        raw = vectorize_bytes(
            data,
            preset_name="logo",
            overrides={"quantize_colors": 4, "boundary_smooth_sigma": 0.0},
        )
        self.assertLessEqual(
            self._fill_count(smoothed.svg), self._fill_count(raw.svg)
        )


class TestPdfExport(unittest.TestCase):
    """The PDF has to be vector too, or the export defeats the point."""

    @classmethod
    def setUpClass(cls):
        cls.outcome = vectorize_bytes(png_bytes(donut()), preset_name="flat_art")
        cls.pdf = svg_to_pdf(cls.outcome.svg)

    def test_is_a_pdf(self):
        self.assertTrue(self.pdf.startswith(b"%PDF-"))

    def test_contains_no_raster_image(self):
        self.assertIsNone(re.search(rb"/Subtype\s*/Image", self.pdf))

    def test_contains_bezier_geometry(self):
        """A PDF of a traced donut should be mostly curve operators."""
        curves = _count_curve_operators(self.pdf)
        self.assertGreater(
            curves, 20, f"only {curves} Bezier operators - is this really vector?"
        )

    def test_page_size_follows_the_artwork(self):
        """902px at 96dpi should become 676.5pt, not a default A4 page."""
        image = Image.new("RGB", (800, 400), (240, 240, 240))
        ImageDraw.Draw(image).ellipse((40, 40, 360, 360), fill=(200, 40, 40))
        outcome = vectorize_bytes(png_bytes(image), preset_name="flat_art")
        pdf = svg_to_pdf(outcome.svg)

        box = re.search(rb"/MediaBox\s*\[\s*([\d.\s]+)\]", pdf)
        self.assertIsNotNone(box, "no MediaBox in the PDF")
        numbers = [float(n) for n in box.group(1).split()]
        width, height = numbers[2] - numbers[0], numbers[3] - numbers[1]
        # 800 x 400 css px -> 600 x 300 pt at the 0.75 css-px-to-point ratio.
        self.assertAlmostEqual(width, 600, delta=2)
        self.assertAlmostEqual(height, 300, delta=2)

    def test_rejects_entity_declarations(self):
        """Billion laughs: lxml expands internal entities."""
        hostile = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE svg [<!ENTITY a "aaaaaaaaaa">]>'
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
            '<path d="M0 0 L5 5 Z"/></svg>'
        )
        with self.assertRaises(InvalidSvgError):
            svg_to_pdf(hostile)

    def test_rejects_external_references(self):
        """svglib resolves href targets, including over the network."""
        hostile = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
            '<image href="http://169.254.169.254/latest/meta-data/" '
            'width="10" height="10"/></svg>'
        )
        with self.assertRaises(InvalidSvgError):
            svg_to_pdf(hostile)

    def test_rejects_oversized_input(self):
        padding = " " * (MAX_SVG_BYTES + 1024)
        with self.assertRaises(InvalidSvgError):
            svg_to_pdf(f'<svg xmlns="http://www.w3.org/2000/svg">{padding}</svg>')

    def test_rejects_non_svg(self):
        with self.assertRaises(InvalidSvgError):
            svg_to_pdf("<html><body>not an svg</body></html>")

    def test_rejects_empty_input(self):
        with self.assertRaises(InvalidSvgError):
            svg_to_pdf("   ")


class TestDeterminism(unittest.TestCase):
    """The same image must always produce the same SVG.

    Regression: `cv2.kmeans` seeds k-means++ from OpenCV's own global RNG,
    which numpy's seeding does not touch. Cluster centres therefore differed
    between runs, so re-exporting the same artwork gave a different file - and
    quality tests comparing two runs were quietly flaky.
    """

    def test_repeated_runs_are_byte_identical(self):
        data = TestCurveQuality.noisy_rings()
        first = vectorize_bytes(data, preset_name="logo").svg
        for attempt in range(3):
            with self.subTest(attempt=attempt):
                self.assertEqual(vectorize_bytes(data, preset_name="logo").svg, first)


class TestSmoothingIsNotGated(unittest.TestCase):
    """Regression: smoothing used to be scaled down by an axis-alignment gate.

    The gate came from a measurement that two later fixes invalidated, and it
    was halving the smoothing on exactly the artwork that needed it - a real
    badge measured 0.42 axis-aligned share and received 0.68 of an intended
    1.40 sigma. Re-measured on the current pipeline, smoothing improves curved
    and rectilinear artwork alike, so the requested sigma is now honoured.
    """

    @staticmethod
    def rectilinear(size: int = 700) -> Image.Image:
        image = Image.new("RGB", (size, size), (245, 243, 235))
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 60, size - 60, 200), fill=(40, 40, 48))
        draw.rectangle((60, 260, 380, 620), fill=(200, 60, 60))
        draw.rectangle((430, 260, size - 60, 430), fill=(60, 110, 190))
        draw.rectangle((430, 470, size - 60, 620), fill=(230, 190, 60))
        return image

    @staticmethod
    def applied_sigma(outcome) -> float:
        for step in outcome.meta["preprocess_steps"]:
            if step.startswith("boundary_smooth"):
                return float(step.split("=")[1].rstrip(")"))
        return 0.0

    def test_requested_sigma_is_applied_to_curved_artwork(self):
        outcome = vectorize_bytes(
            TestCurveQuality.noisy_rings(),
            preset_name="logo",
            overrides={"boundary_smooth_sigma": 0.7},
        )
        # 0.7 output pixels at 2x supersampling.
        self.assertAlmostEqual(self.applied_sigma(outcome), 1.4, places=2)

    def test_requested_sigma_is_applied_to_rectilinear_artwork_too(self):
        buffer = io.BytesIO()
        self.rectilinear().save(buffer, format="JPEG", quality=85)
        outcome = vectorize_bytes(
            buffer.getvalue(),
            preset_name="logo",
            overrides={"boundary_smooth_sigma": 0.7},
        )
        self.assertAlmostEqual(self.applied_sigma(outcome), 1.4, places=2)

    def test_ineffective_sigmas_are_treated_as_off(self):
        """A sigma too small to flip any vote should not cost a blur pass."""
        outcome = vectorize_bytes(
            TestCurveQuality.noisy_rings(),
            preset_name="logo",
            overrides={"boundary_smooth_sigma": 0.2},  # 0.4 applied
        )
        self.assertEqual(self.applied_sigma(outcome), 0.0)


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
