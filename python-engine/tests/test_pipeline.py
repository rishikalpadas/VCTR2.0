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
from image_io import load_image  # noqa: E402
from preprocessing import (  # noqa: E402
    _dissolve_edge_films,
    _unblend_line_edges,
    preprocess,
)
from presets import apply_overrides, get_preset  # noqa: E402
from svg_export import (  # noqa: E402
    MAX_SVG_BYTES,
    _count_curve_operators,
    svg_to_pdf,
)
from svg_optimizer import validate_svg  # noqa: E402
from vectorizer import vectorize_bytes  # noqa: E402
from engines.potrace_engine import _tuck_fills_under_ink  # noqa: E402

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
            # An explicit k is the cluster count, not a guarantee about the
            # surviving palette: the blend filter runs afterwards and removes
            # entries that are only the average of two others, which on a
            # two-colour donut is exactly what the surplus clusters land on.
            # Disabled here so this keeps testing the override itself.
            overrides={"quantize_colors": 5, "drop_blend_colors": False},
        )
        self.assertIn("quantize(k=5)", outcome.meta["preprocess_steps"])

    def test_surplus_clusters_are_dropped_as_blends(self):
        """Asking for more colours than the artwork has must not invent any:
        the extra clusters land on anti-aliased edges, and those are averages
        of two real colours, not colours of their own."""
        outcome = vectorize_bytes(
            png_bytes(donut()),
            preset_name="flat_art",
            overrides={"quantize_colors": 5},
        )
        self.assertNotIn("quantize(k=5)", outcome.meta["preprocess_steps"])
        self.assertLess(len(outcome.meta["palette"]), 5)


KEYLINE_SIZE = 2400
KEYLINE_RULE_Y = KEYLINE_SIZE - KEYLINE_SIZE // 15


def keylined_art() -> Image.Image:
    """Flat colour rings separated by thin dark keylines, plus a hairline rule.

    Deliberately the shape of real sticker artwork, and deliberately large: the
    strokes are 2px at 2400px, but the logo preset works at 1100px, so by the
    time the tracer sees them they are under a pixel wide. Drawn oversized and
    downsampled so every stroke carries a genuine anti-aliasing ramp - a
    hard-edged synthetic stroke survives resampling that a real one does not,
    which is what made an earlier version of this fixture prove nothing.
    """
    supersize, stroke = 2, 2
    canvas = KEYLINE_SIZE * supersize
    image = Image.new("RGB", (canvas, canvas), (252, 252, 254))
    draw = ImageDraw.Draw(image)
    for index, fill in enumerate([(172, 85, 54), (245, 168, 131), (253, 219, 126)]):
        inset = canvas * (8 + index * 9) // 100
        draw.ellipse(
            (inset, inset, canvas - inset, canvas - inset),
            fill=fill,
            outline=(40, 40, 40),
            width=stroke * supersize,
        )
    rule_y = KEYLINE_RULE_Y * supersize
    draw.line(
        (canvas // 6, rule_y, canvas - canvas // 6, rule_y),
        fill=(40, 40, 40),
        width=stroke * supersize,
    )
    return image.resize((KEYLINE_SIZE, KEYLINE_SIZE), Image.LANCZOS)


class TestLinework(unittest.TestCase):
    """Thin dark strokes must survive the downscale-and-smooth pipeline."""

    @staticmethod
    def prepare(image: Image.Image, **overrides):
        rgba, info = load_image(png_bytes(image))
        preset = apply_overrides(
            get_preset("logo"), {"background": "never", **overrides}
        )
        return preprocess(
            rgba, preset.preprocess, analyze(rgba), source_format=info["format"]
        )

    @staticmethod
    def dark_pixels(rgba: np.ndarray) -> int:
        luma = rgba[..., :3].astype(np.float32) @ np.array(
            [0.299, 0.587, 0.114], dtype=np.float32
        )
        return int((luma <= 100).sum())

    @staticmethod
    def darkest_along_rule(prepared) -> float:
        """Luma of the darkest pixel across the hairline rule.

        The rule is the cleanest witness for the whole class of bug: it is dark
        on white, so when the pipeline loses it the pixels do not disappear,
        they get reassigned to the nearest surviving palette entry - which is a
        pale one. A pale reading here is the "the rule came back light blue"
        report, reduced to a number.
        """
        y = round(KEYLINE_RULE_Y * prepared.scale)
        x = round(KEYLINE_SIZE * 0.5 * prepared.scale)
        band = prepared.image[y - 6 : y + 7, x - 40 : x + 41, :3].astype(np.float32)
        return float((band @ np.array([0.299, 0.587, 0.114], np.float32)).min())

    def test_keylines_survive_the_downscale(self):
        art = keylined_art()
        without = self.prepare(art, preserve_linework=False)
        with_ink = self.prepare(art, preserve_linework=True)

        self.assertIn("linework_restored", " ".join(with_ink.steps))
        self.assertGreater(
            self.dark_pixels(with_ink.image), self.dark_pixels(without.image) * 1.4
        )

    def test_hairline_rule_keeps_its_colour(self):
        art = keylined_art()
        # Without the fix the rule survives geometrically but lands on a pale
        # palette entry; with it, ink stays ink.
        self.assertGreater(
            self.darkest_along_rule(self.prepare(art, preserve_linework=False)), 120
        )
        self.assertLess(
            self.darkest_along_rule(self.prepare(art, preserve_linework=True)), 100
        )

    @staticmethod
    def ring(width: int, size: int = 900) -> Image.Image:
        image = Image.new("RGB", (size, size), (250, 250, 252))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (120, 120, size - 120, size - 120), outline=(40, 40, 40), width=width
        )
        return image

    def test_thick_dark_shapes_are_left_to_the_smoother(self):
        """Only strokes thin enough to be lost get stamped back.

        A dark shape wide enough to survive quantization has already had its
        boundary smoothed by the majority vote; stamping the raw source mask
        over it reinstates exactly the pixel wobble that smoothing removed.
        When this broke, boundary smoothing became a no-op on dark artwork -
        identical segment counts with it on and off.
        """
        thin = self.prepare(self.ring(3), preserve_linework=True)
        thick = self.prepare(self.ring(24), preserve_linework=True)
        self.assertIn("linework_restored", " ".join(thin.steps))
        self.assertNotIn("linework_restored", " ".join(thick.steps))

    def test_potrace_paints_the_ink_layer_last(self):
        """Line work has to win the seam it shares with every fill.

        Each colour layer is traced independently, so a shared boundary comes
        back as two slightly different curves and whichever is painted later
        wins. With the ink painted first, every fill bulged into it by however
        much its own curve fit happened to differ - the stroke reads thinner in
        some places, and the fill that ate it shows as colour inside the stroke.
        """
        outcome = vectorize_bytes(
            png_bytes(keylined_art()),
            preset_name="logo",
            overrides={"background": "never", "engine": "potrace"},
        )
        order = [layer["color"] for layer in outcome.meta["engine_meta"]["layers"]]
        luma = lambda h: (  # noqa: E731
            0.299 * int(h[1:3], 16) + 0.587 * int(h[3:5], 16) + 0.114 * int(h[5:7], 16)
        )
        self.assertGreater(len(order), 1)
        self.assertEqual(order[-1], min(order, key=luma), f"paint order {order}")

    def test_restored_ink_stays_on_the_palette(self):
        """Stamping must not introduce a colour, or it becomes an extra layer."""
        prepared = self.prepare(keylined_art(), preserve_linework=True)
        visible = prepared.image[prepared.image[..., 3] > 0][:, :3]
        present = {"#{:02x}{:02x}{:02x}".format(*map(int, c)) for c in np.unique(visible, axis=0)}
        self.assertTrue(present <= set(prepared.palette), present - set(prepared.palette))


class TestEdgeFilms(unittest.TestCase):
    """The hairline of a wrong colour that anti-aliasing leaves on a stroke.

    A stroke's anti-aliased edge is a blend of the ink and whatever it runs
    over, and quantization has to put that blend on some palette entry. When a
    real colour of the artwork sits near the blend in RGB it collects the whole
    edge, and every stroke comes back wrapped in a hairline of a colour that is
    nowhere near it in the drawing.
    """

    INK, PALE, PURPLE = (39, 39, 39), (208, 229, 230), (167, 112, 194)
    PALETTE = np.array([INK, PALE, PURPLE], dtype=np.uint8)

    def art(self, film=PURPLE):
        """Ink bar, one row of ``film`` under it, pale below, real purple away."""
        image = np.zeros((40, 40, 4), dtype=np.uint8)
        image[..., 3] = 255
        image[:20, :, :3] = self.INK
        image[20, :, :3] = film
        image[21:, :, :3] = self.PALE
        image[30:, 30:, :3] = self.PURPLE
        original = image[..., :3].copy()
        original[20] = (176, 196, 197)  # the blend it really was: nearer pale
        return image, original

    def test_a_film_against_the_ink_is_reassigned(self):
        image, original = self.art()
        out, changed = _dissolve_edge_films(image, self.PALETTE, original)
        self.assertEqual(changed, 40)
        np.testing.assert_array_equal(out[20, :, :3], np.tile(self.PALE, (40, 1)))

    def test_it_resolves_to_the_side_the_pixel_came_from(self):
        """Nearer the ink in the original, so it belongs to the stroke."""
        image, original = self.art()
        original[20] = (70, 70, 70)
        out, _ = _dissolve_edge_films(image, self.PALETTE, original)
        np.testing.assert_array_equal(out[20, :, :3], np.tile(self.INK, (40, 1)))

    def test_a_real_region_of_the_same_colour_survives(self):
        """The colour is not the evidence - being a hairline on ink is."""
        image, original = self.art()
        out, _ = _dissolve_edge_films(image, self.PALETTE, original)
        np.testing.assert_array_equal(out[30:, 30:, :3], image[30:, 30:, :3])

    def test_a_hairline_of_ink_is_never_dissolved(self):
        """Thin dark strokes are the one thing that really is a pixel wide."""
        image = np.zeros((40, 40, 4), dtype=np.uint8)
        image[..., 3] = 255
        image[:, :, :3] = self.PALE
        image[20, :, :3] = self.INK
        out, changed = _dissolve_edge_films(image, self.PALETTE, image[..., :3].copy())
        self.assertEqual(changed, 0)
        np.testing.assert_array_equal(out[20, :, :3], np.tile(self.INK, (40, 1)))

    def test_the_result_holds_palette_colours_only(self):
        """Each stray blend left behind costs Potrace a layer and a subprocess."""
        image, original = self.art(film=(123, 134, 135))
        out, _ = _dissolve_edge_films(image, self.PALETTE, original)
        present = np.unique(out[..., :3].reshape(-1, 3), axis=0)
        for colour in present.tolist():
            self.assertIn(colour, self.PALETTE.tolist(), f"{colour} is off-palette")


class TestInkTuck(unittest.TestCase):
    """Fill layers run on underneath the line work instead of stopping at it.

    Potrace fits every colour layer's outline on its own mask. A fill that
    stops exactly at the ink therefore has an outline that lands on the ink's
    edge, and wherever its curve fit bulges by a fraction of a pixel it pokes
    out past the stroke - colour leaking into the neighbouring region - or
    falls short of it, leaving a background sliver. Tucked under the ink, the
    fill's own outline sits in the middle of the stroke, where the ink painted
    on top hides whatever its curve fit does.
    """

    INK = "#272727"

    @staticmethod
    def banded(width: int = 40) -> list[tuple[str, np.ndarray]]:
        """Red | 4px ink stroke | blue, left to right; ink traced last."""
        red = np.zeros((20, width), bool)
        blue = np.zeros((20, width), bool)
        ink = np.zeros((20, width), bool)
        red[:, :18] = True
        ink[:, 18:22] = True
        blue[:, 22:] = True
        return [("#ff0000", red), ("#0000ff", blue), (TestInkTuck.INK, ink)]

    def test_fills_split_the_stroke_between_them(self):
        red, blue, ink = _tuck_fills_under_ink(self.banded(), reach=4, speck_area=0)
        self.assertTrue(red[1][:, :20].all(), "red stops short of the stroke centre")
        self.assertTrue(blue[1][:, 20:].all(), "blue stops short of the stroke centre")
        self.assertFalse((red[1] & blue[1]).any(), "fills overlap each other")

    def test_the_ink_itself_is_untouched(self):
        layers = self.banded()
        tucked = _tuck_fills_under_ink(layers, reach=4, speck_area=0)
        self.assertEqual(tucked[-1][0], self.INK)
        np.testing.assert_array_equal(tucked[-1][1], layers[-1][1])

    def test_reach_limits_how_far_a_fill_runs_under_ink(self):
        """A dark backdrop merged with the ink must not become a giant fill."""
        red = np.zeros((20, 60), bool)
        ink = np.ones((20, 60), bool)
        red[:, :10] = True
        ink[:, :10] = False
        tucked = _tuck_fills_under_ink(
            [("#ff0000", red), (self.INK, ink)], reach=3, speck_area=0
        )
        self.assertTrue(tucked[0][1][:, :13].all())
        self.assertFalse(tucked[0][1][:, 14:].any())

    def test_transparent_pixels_stay_unfilled(self):
        """Ink against a removed background splits with nothing, not a fill."""
        red = np.zeros((20, 40), bool)
        ink = np.zeros((20, 40), bool)
        red[:, :18] = True
        ink[:, 18:22] = True  # everything right of the stroke is transparent
        tucked = _tuck_fills_under_ink(
            [("#ff0000", red), (self.INK, ink)], reach=4, speck_area=0
        )
        self.assertFalse(tucked[0][1][:, 20:].any(), "fill ran out past the stroke")

    def test_specks_are_merged_into_their_neighbour(self):
        """A few pixels of purple stranded against a stroke is not artwork."""
        layers = self.banded()
        purple = np.zeros((20, 40), bool)
        purple[8:10, 23:25] = True
        blue = layers[1][1] & ~purple
        layers = [layers[0], ("#0000ff", blue), ("#aa66cc", purple), layers[2]]
        tucked = dict(_tuck_fills_under_ink(layers, reach=4, speck_area=10))
        self.assertNotIn("#aa66cc", tucked, "the speck survived as its own layer")
        self.assertTrue(tucked["#0000ff"][8:10, 23:25].all())

    def test_regions_above_the_speck_size_survive(self):
        layers = self.banded()
        purple = np.zeros((20, 40), bool)
        purple[4:12, 26:34] = True
        layers = [layers[0], ("#0000ff", layers[1][1] & ~purple), ("#aa66cc", purple), layers[2]]
        tucked = dict(_tuck_fills_under_ink(layers, reach=4, speck_area=10))
        self.assertTrue(tucked["#aa66cc"][4:12, 26:34].all())

    def test_a_single_layer_is_returned_as_is(self):
        only = [("#ff0000", np.ones((4, 4), bool))]
        self.assertIs(_tuck_fills_under_ink(only, reach=4, speck_area=10), only)

    def test_no_seam_between_fill_and_ink_end_to_end(self):
        """Every opaque pixel of the keylined art is covered by some layer."""
        outcome = vectorize_bytes(
            png_bytes(keylined_art()),
            preset_name="logo",
            overrides={"background": "never", "engine": "potrace"},
        )
        self.assertNotIn("stroke=", outcome.svg, "the old stroked underlay is back")
        fills = re.findall(r'<path[^>]*?fill="(#[0-9a-fA-F]{6})"', outcome.svg)
        luma = [0.299 * int(f[1:3], 16) + 0.587 * int(f[3:5], 16) + 0.114 * int(f[5:7], 16) for f in fills]
        self.assertEqual(luma[-1], min(luma), "line work is not painted last")


class TestUnblendLineEdges(unittest.TestCase):
    """Anti-aliased stroke edges go to the fill they were blended from.

    A pixel half way between a pale fill and the ink averages out to a mid
    colour, and plain nearest-colour quantization hands it to whichever *third*
    palette entry happens to sit there - a blue notch eaten out of the corner
    of a pale-blue pole, a purple hairline along a pale-blue stroke.
    """

    PALETTE = np.array([[39, 39, 39], [208, 229, 230], [77, 145, 194]], np.uint8)

    def image(self):
        """Pale | ink | pale, with the pale-to-ink edge column quantized blue."""
        rgba = np.zeros((10, 12, 4), np.uint8)
        rgba[..., 3] = 255
        rgba[..., :3] = self.PALETTE[1]
        rgba[:, 5:7, :3] = self.PALETTE[0]
        original = rgba[..., :3].copy()
        mix = (0.6 * self.PALETTE[1] + 0.4 * self.PALETTE[0]).astype(np.uint8)
        original[:, 4] = mix
        rgba[:, 4, :3] = self.PALETTE[2]  # what nearest-colour made of the blend
        rgba[0, 11, :3] = self.PALETTE[2]  # blue really is in this artwork
        return rgba, original

    def test_a_blend_goes_back_to_the_fill_it_came_from(self):
        rgba, original = self.image()
        out, changed = _unblend_line_edges(rgba, self.PALETTE, original, reach=2)
        self.assertEqual(changed, 10)
        self.assertTrue((out[:, 4, :3] == self.PALETTE[1]).all())

    def test_a_real_region_of_the_third_colour_is_left_alone(self):
        rgba, original = self.image()
        rgba[:, 0:3, :3] = self.PALETTE[2]
        original[:, 0:3] = self.PALETTE[2]
        out, _ = _unblend_line_edges(rgba, self.PALETTE, original, reach=2)
        self.assertTrue((out[:, 0:3, :3] == self.PALETTE[2]).all())

    def test_the_ink_is_never_moved(self):
        rgba, original = self.image()
        out, _ = _unblend_line_edges(rgba, self.PALETTE, original, reach=2)
        self.assertTrue((out[:, 5:7, :3] == self.PALETTE[0]).all())

    def test_the_ink_never_grows(self):
        """Even a mostly-ink blend stays a fill: stroke weight is not ours to set."""
        rgba, original = self.image()
        original[:, 4] = (0.2 * self.PALETTE[1] + 0.8 * self.PALETTE[0]).astype(np.uint8)
        out, _ = _unblend_line_edges(rgba, self.PALETTE, original, reach=2)
        self.assertTrue((out[:, 4, :3] == self.PALETTE[1]).all())


def small_pale_region(size: int = 1200) -> Image.Image:
    """White field, a big blue block, and a small pale-blue shape inside it.

    The pale blue is a real colour of the artwork, but it is small and sits
    almost exactly on the line between the white and the blue in RGB - which is
    what an anti-aliasing blend between those two looks like as well. Measured
    on real sticker artwork, the pale blue was *more* collinear than the actual
    halo, so a colour-geometry test alone flattens it to white.
    """
    image = Image.new("RGB", (size, size), (252, 253, 254))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (size // 8, size // 8, size * 7 // 8, size * 7 // 8), fill=(77, 147, 196)
    )
    draw.rectangle(
        (size * 2 // 5, size * 2 // 5, size * 3 // 5, size * 3 // 5),
        fill=(209, 229, 231),
    )
    return image


class TestBlendColours(unittest.TestCase):
    def test_a_small_real_colour_is_not_mistaken_for_a_blend(self):
        outcome = vectorize_bytes(
            png_bytes(small_pale_region()),
            preset_name="flat_art",
            overrides={"background": "never"},
        )
        wanted = np.array([209, 229, 231], dtype=np.float32)
        closest = min(
            np.linalg.norm(
                np.array([int(entry[i : i + 2], 16) for i in (1, 3, 5)], np.float32)
                - wanted
            )
            for entry in outcome.meta["palette"]
        )
        self.assertLess(closest, 25, f"pale blue lost; palette {outcome.meta['palette']}")


class TestPaletteIsClosed(unittest.TestCase):
    def test_every_fill_comes_from_the_quantized_palette(self):
        """VTracer averages each layer's colour, so one yellow in the artwork
        comes back as four near-identical yellows unless they are snapped."""
        outcome = vectorize_bytes(
            png_bytes(keylined_art()),
            preset_name="flat_art",
            overrides={"background": "never", "engine": "vtracer"},
        )
        palette = set(outcome.meta["palette"])
        fills = {f.lower() for f in re.findall(r'fill="(#[0-9a-fA-F]{6})"', outcome.svg)}
        self.assertTrue(fills - palette == set(), f"off-palette fills: {fills - palette}")

    def test_snapping_is_skipped_when_there_is_no_palette(self):
        outcome = vectorize_bytes(
            png_bytes(donut()),
            preset_name="flat_art",
            overrides={"quantize_colors": None},
        )
        self.assertIsNone(outcome.meta["palette"])
        self.assertEqual(outcome.meta["snapped_fills"], 0)


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
