"""Runtime configuration for the vectorization service.

Values can be overridden with environment variables (or a local .env file),
e.g. ``VEC_MAX_UPLOAD_BYTES=20971520``.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # --- upload limits -----------------------------------------------------
    max_upload_bytes: int = 15 * 1024 * 1024          # 15 MB
    max_input_pixels: int = 40_000_000                # decompression-bomb guard
    min_dimension: int = 8                            # reject 1x1 / garbage

    # Formats we accept. Keys are magic-byte-derived, not filename-derived.
    allowed_formats: tuple[str, ...] = ("png", "jpeg", "webp")

    # --- pipeline safety ---------------------------------------------------
    # Hard ceiling applied *after* the preset's own max_dimension.
    # Ceiling on the longest edge actually handed to the tracer, applied after
    # the preset max_dimension AND its supersampling factor. Supersampling is
    # what needs the headroom: a 1200px preset at 2x wants 2400px.
    absolute_max_dimension: int = 3000
    # Paths whose bounding box is smaller than this fraction of the canvas
    # diagonal get dropped as tracing noise.
    default_min_path_diagonal_ratio: float = 0.004


settings = Settings()
