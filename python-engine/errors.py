"""Typed pipeline errors.

Every error carries a stable machine-readable ``code`` so the Node layer can
map it to an HTTP status and a user-facing message without ever forwarding a
Python traceback to the browser.
"""

from __future__ import annotations


class VectorizationError(Exception):
    """Base class for all errors raised by the vectorization pipeline."""

    code = "VECTORIZATION_FAILED"
    http_status = 500

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_payload(self) -> dict:
        return {"code": self.code, "message": self.message}


class UnsupportedFormatError(VectorizationError):
    code = "UNSUPPORTED_FORMAT"
    http_status = 415


class CorruptImageError(VectorizationError):
    code = "CORRUPT_IMAGE"
    http_status = 400


class ImageTooLargeError(VectorizationError):
    code = "IMAGE_TOO_LARGE"
    http_status = 413


class ImageTooSmallError(VectorizationError):
    code = "IMAGE_TOO_SMALL"
    http_status = 400


class UnknownPresetError(VectorizationError):
    code = "UNKNOWN_PRESET"
    http_status = 400


class EngineError(VectorizationError):
    code = "ENGINE_FAILED"
    http_status = 500


class EngineNotImplementedError(VectorizationError):
    code = "ENGINE_NOT_IMPLEMENTED"
    http_status = 501


class InvalidSvgError(VectorizationError):
    code = "INVALID_SVG"
    http_status = 500
