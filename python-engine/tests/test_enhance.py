"""Tests for enhance.py - the AI upscale of soft / low-resolution uploads.

Run from the python-engine directory:

    venv\\Scripts\\python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import enhance  # noqa: E402
from config import settings  # noqa: E402
from vectorizer import vectorize_bytes  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
POTRACE = {"engine": "potrace"}


def badge(size: int = 900) -> np.ndarray:
    """Sharp flat artwork: a ring, a bar and a dark rule on cream."""
    image = Image.new("RGB", (size, size), (245, 240, 230))
    draw = ImageDraw.Draw(image)
    s = size / 900
    draw.ellipse((90 * s, 90 * s, 810 * s, 810 * s), fill=(200, 50, 60))
    draw.ellipse((240 * s, 240 * s, 660 * s, 660 * s), fill=(245, 240, 230))
    draw.rectangle((150 * s, 410 * s, 750 * s, 490 * s), fill=(30, 70, 140))
    draw.line((120 * s, 840 * s, 780 * s, 840 * s), fill=(25, 25, 30), width=max(1, round(8 * s)))
    rgb = np.array(image)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def blown_up(rgba: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """What a customer uploads: a small copy stretched back up, JPEG-soft."""
    h, w = rgba.shape[:2]
    small = cv2.resize(rgba, (round(w / factor), round(h / factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def encode(rgba: np.ndarray, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(rgba[..., :3] if fmt == "JPEG" else rgba)
    image.save(buffer, format=fmt, **({"quality": 70} if fmt == "JPEG" else {}))
    return buffer.getvalue()


@unittest.skipUnless(enhance.available(), "super-resolution models not installed")
class TestBlurMeasure(unittest.TestCase):
    def test_sharp_artwork_measures_sharp(self):
        blur = enhance.measure_blur(badge())
        self.assertLess(blur["blur_factor"], 1.4)

    def test_known_blur_is_recovered(self):
        """Calibration check: a 1.3px Gaussian blur on sharp art reads as ~1.3px."""
        image = badge()
        image[..., :3] = cv2.GaussianBlur(image[..., :3], (0, 0), 1.3)
        sigma = enhance.measure_blur(image)["blur_sigma"]
        self.assertAlmostEqual(sigma, 1.35, delta=0.25)

    def test_blown_up_image_reports_its_real_resolution(self):
        """A 300px image stretched to 900px holds roughly 300-450px of detail."""
        blur = enhance.measure_blur(blown_up(badge(), 3.0))
        self.assertGreaterEqual(blur["blur_factor"], 2.0)
        self.assertLess(blur["effective_size"], 450)

    def test_flat_image_without_edges_is_not_judged(self):
        flat = np.full((300, 300, 4), 200, np.uint8)
        self.assertIsNone(enhance.measure_blur(flat)["blur_sigma"])


@unittest.skipUnless(enhance.available(), "super-resolution models not installed")
class TestGate(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.object(settings, "enhance", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sharp_large_image_is_left_alone(self):
        source = badge()
        result = enhance.enhance(source)
        self.assertFalse(result.applied)
        self.assertIs(result.image, source)

    def test_blurry_image_is_enhanced(self):
        result = enhance.enhance(blown_up(badge()))
        self.assertTrue(result.applied)
        self.assertEqual(result.report["model"], enhance.FLAT_MODEL)

    def test_small_sharp_image_is_enhanced(self):
        result = enhance.enhance(badge(300))
        self.assertTrue(result.applied)
        self.assertEqual(result.image.shape[:2], (1200, 1200))

    def test_request_can_force_either_way(self):
        self.assertTrue(enhance.enhance(badge(), mode="always").applied)
        self.assertFalse(enhance.enhance(blown_up(badge()), mode="never").applied)

    def test_server_switch_turns_it_off(self):
        with unittest.mock.patch.object(settings, "enhance", False):
            self.assertFalse(enhance.enhance(blown_up(badge()), mode="always").applied)

    def test_photographs_use_the_photo_model(self):
        result = enhance.enhance(blown_up(badge()), photographic=True)
        self.assertEqual(result.report["model"], enhance.PHOTO_MODEL)

    def test_a_broken_model_never_costs_the_conversion(self):
        source = blown_up(badge())
        with unittest.mock.patch.object(enhance, "_upscale", side_effect=RuntimeError("boom")):
            result = enhance.enhance(source)
        self.assertFalse(result.applied)
        self.assertIs(result.image, source)

    def test_parse_mode(self):
        self.assertEqual(enhance.parse_mode(None), "auto")
        self.assertEqual(enhance.parse_mode("never"), "never")
        self.assertEqual(enhance.parse_mode(True), "always")
        self.assertEqual(enhance.parse_mode(False), "never")
        self.assertEqual(enhance.parse_mode("bogus"), "auto")


@unittest.skipUnless(enhance.available(), "super-resolution models not installed")
class TestOutput(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.object(settings, "enhance", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_edges_come_back_sharper(self):
        source = blown_up(badge())
        result = enhance.enhance(source)
        # Compared at the same scale: blur is measured in pixels.
        same_scale = cv2.resize(result.image, source.shape[1::-1], interpolation=cv2.INTER_AREA)
        before = enhance.measure_blur(source)["blur_sigma"]
        after = enhance.measure_blur(same_scale)["blur_sigma"]
        self.assertLess(after, before * 0.7)  # measured 1.38 -> 0.87

    def test_flat_colours_stay_the_source_colours(self):
        """Colour fidelity is a hard requirement: flat regions keep their colour."""
        source = blown_up(badge())
        out = enhance.enhance(source).image
        out = cv2.resize(out, source.shape[1::-1], interpolation=cv2.INTER_AREA)
        for name, (y, x) in {"ring": (450, 150), "bar": (450, 450), "cream": (40, 40)}.items():
            with self.subTest(region=name):
                want = source[y - 10:y + 10, x - 10:x + 10, :3].reshape(-1, 3).mean(0)
                got = out[y - 10:y + 10, x - 10:x + 10, :3].reshape(-1, 3).mean(0)
                self.assertLess(float(np.abs(want - got).max()), 4.0)

    def test_transparency_survives_without_a_dark_fringe(self):
        art = badge(300)
        art[..., 3] = 0
        cv2.circle(art, (150, 150), 100, (230, 200, 40, 255), -1)
        art[art[..., 3] == 0, :3] = 0  # transparent pixels stored as black
        out = enhance.enhance(art).image
        self.assertEqual(out.shape[2], 4)
        self.assertEqual(int(out[10, 10, 3]), 0)
        self.assertEqual(int(out[600, 600, 3]), 255)
        # Just inside the edge of the disc: no black bled in from outside.
        edge = out[600, 600 - 395:600 - 385, :3].astype(int)
        self.assertGreater(int(edge.sum(axis=1).min()), 300)

    def test_output_stays_under_the_size_cap(self):
        source = blown_up(badge(2400), 1.3)
        result = enhance.enhance(source, mode="always")
        self.assertLessEqual(max(result.image.shape[:2]), settings.absolute_max_dimension)


@unittest.skipUnless(enhance.available(), "super-resolution models not installed")
class TestPipeline(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.object(settings, "enhance", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_blurry_upload_traces_cleaner_with_potrace(self):
        data = encode(blown_up(badge()), "JPEG")
        off = vectorize_bytes(data, preset_name="logo", overrides={**POTRACE, "enhance": "never"})
        on = vectorize_bytes(data, preset_name="logo", overrides=POTRACE)
        self.assertTrue(on.meta["enhance"]["applied"])
        self.assertEqual(on.meta["engine"], "potrace")
        # No colour the unenhanced trace did not already need, and less
        # geometry to draw the same four shapes.
        self.assertLessEqual(len(on.meta["palette"]), len(off.meta["palette"]))
        self.assertLess(on.meta["path_count"], off.meta["path_count"])
        self.assertLess(on.meta["svg_bytes"], off.meta["svg_bytes"])

    def test_svg_keeps_the_upload_dimensions(self):
        on = vectorize_bytes(encode(badge(300)), preset_name="logo", overrides=POTRACE)
        self.assertTrue(on.meta["enhance"]["applied"])
        self.assertIn('width="300"', on.svg)
        self.assertIn('height="300"', on.svg)

    def test_enhanced_trace_keeps_the_preset_supersample(self):
        """Output-pixel settings keep their meaning: the 2x preset still reports 2x."""
        on = vectorize_bytes(encode(blown_up(badge()), "JPEG"), preset_name="logo", overrides=POTRACE)
        self.assertEqual(on.meta["supersample"], 2.0)
        self.assertTrue(any(s.startswith("enhanced_fit") for s in on.meta["preprocess_steps"]))
        self.assertNotIn("jpeg_cleanup", on.meta["preprocess_steps"])

    def test_approved_sharp_artwork_is_untouched(self):
        """The locked WEAKNESS sticker is sharp and large: the gate must skip it."""
        source = (FIXTURES / "weakness.jpeg").read_bytes()
        outcome = vectorize_bytes(
            source, preset_name="auto", overrides={"background": "never", "engine": "potrace"}
        )
        self.assertFalse(outcome.meta["enhance"]["applied"])
        expected = (FIXTURES / "weakness-potrace.svg").read_text(encoding="utf-8")
        self.assertEqual(outcome.svg, expected)


if __name__ == "__main__":
    unittest.main()
