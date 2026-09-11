"""Safe image loading.

The format is decided by inspecting the bytes, never by trusting a filename or
a client-supplied Content-Type.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import settings
from errors import (
    CorruptImageError,
    ImageTooLargeError,
    ImageTooSmallError,
    UnsupportedFormatError,
)
from logging_config import get_logger

log = get_logger(__name__)

# Pillow refuses absurd images itself, but set our own ceiling explicitly.
Image.MAX_IMAGE_PIXELS = settings.max_input_pixels

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
)


def sniff_format(data: bytes) -> str | None:
    """Return a lowercase format name from the leading bytes, or None."""
    for signature, name in _MAGIC:
        if data.startswith(signature):
            return name
    # WebP: "RIFF" .... "WEBP"
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def load_image(data: bytes) -> tuple[np.ndarray, dict]:
    """Decode ``data`` into an RGBA uint8 array.

    Returns ``(rgba_array, info)`` where ``info`` records the detected format
    and the original pixel dimensions.
    """
    if not data:
        raise CorruptImageError("Uploaded file is empty.")

    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"File is larger than the "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB limit."
        )

    fmt = sniff_format(data)
    if fmt is None or fmt not in settings.allowed_formats:
        raise UnsupportedFormatError(
            "Unsupported image format. Upload a PNG, JPG/JPEG or WebP file."
        )

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            # Honour the EXIF orientation flag so phone photos are upright.
            img = ImageOps.exif_transpose(img)
            width, height = img.size

            if width * height > settings.max_input_pixels:
                raise ImageTooLargeError(
                    f"Image is {width}x{height}px, which exceeds the "
                    f"{settings.max_input_pixels // 1_000_000} megapixel limit."
                )
            if min(width, height) < settings.min_dimension:
                raise ImageTooSmallError(
                    f"Image is only {width}x{height}px - too small to vectorize."
                )

            rgba = img.convert("RGBA")
            array = np.array(rgba, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # OSError covers truncated files; log the real reason, tell the user
        # something useful but generic.
        log.warning("Image decode failed: %s: %s", type(exc).__name__, exc)
        raise CorruptImageError(
            "The image could not be decoded. It may be corrupted or truncated."
        ) from exc

    info = {
        "format": fmt,
        "original_width": int(array.shape[1]),
        "original_height": int(array.shape[0]),
        "bytes": len(data),
    }
    log.info(
        "Loaded %s %dx%d (%.1f KB)",
        fmt,
        info["original_width"],
        info["original_height"],
        len(data) / 1024,
    )
    return array, info


def to_png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an RGBA array as PNG bytes (the format handed to the engine)."""
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()
