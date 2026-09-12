"""FastAPI vectorization service.

Runs standalone on http://127.0.0.1:8000 and is the only place image
processing happens. The Node API is a thin proxy in front of it, which is what
lets this service move to its own (possibly beefier) host later without the
MERN app knowing.

Start with:  python app.py
"""

from __future__ import annotations

import json
import time

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from config import settings
from errors import ImageTooLargeError, InvalidSvgError, VectorizationError
from logging_config import configure_logging, get_logger
from presets import list_presets
from svg_export import svg_to_pdf
from schemas import (
    EngineListResponse,
    ErrorResponse,
    HealthResponse,
    PresetListResponse,
    VectorizeResponse,
)
from vectorizer import describe_engines, vectorize_bytes

VERSION = "1.0.0"

configure_logging()
log = get_logger("app")

app = FastAPI(
    title="Image-to-Vector Engine",
    version=VERSION,
    description=(
        "CPU raster-to-vector service. Converts clean digital artwork into "
        "real SVG path geometry."
    ),
)


# ---------------------------------------------------------------------------
# Error handling: log everything, leak nothing
# ---------------------------------------------------------------------------


@app.exception_handler(VectorizationError)
async def vectorization_error_handler(request: Request, exc: VectorizationError):
    log.warning(
        "%s on %s: %s%s",
        exc.code,
        request.url.path,
        exc.message,
        f" | detail: {exc.detail}" if exc.detail else "",
    )
    return JSONResponse(
        status_code=exc.http_status,
        content=ErrorResponse(error={"code": exc.code, "message": exc.message}).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    # Full traceback to the server log; a generic message to the caller.
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred while processing the image.",
            }
        ).model_dump(),
    )


@app.middleware("http")
async def access_log(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    log.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=VERSION, engines=describe_engines())


@app.get("/presets", response_model=PresetListResponse)
async def presets() -> PresetListResponse:
    return PresetListResponse(presets=list_presets())


@app.get("/engines", response_model=EngineListResponse)
async def engines() -> EngineListResponse:
    return EngineListResponse(engines=describe_engines())


@app.post(
    "/vectorize",
    response_model=VectorizeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def vectorize(
    file: UploadFile = File(..., description="PNG, JPG/JPEG or WebP image"),
    preset: str = Form("auto"),
    options: str | None = Form(None),
) -> VectorizeResponse:
    """Convert one raster image into SVG vector paths.

    ``options`` is an optional JSON object with a narrow set of overrides -
    see ``presets.apply_overrides``.
    """
    data = await file.read()

    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"File is larger than the "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB limit."
        )

    overrides: dict | None = None
    if options:
        try:
            parsed = json.loads(options)
            if isinstance(parsed, dict):
                overrides = parsed
        except json.JSONDecodeError:
            log.warning("Ignoring malformed options JSON: %r", options[:200])

    outcome = vectorize_bytes(data, preset_name=preset, overrides=overrides)
    return VectorizeResponse(svg=outcome.svg, meta=outcome.meta)


@app.post(
    "/export/pdf",
    responses={
        200: {"content": {"application/pdf": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def export_pdf(file: UploadFile = File(..., description="SVG document")):
    """Convert a generated SVG into a single-page vector PDF.

    Takes the SVG rather than re-tracing the original raster: the export then
    matches exactly what the user previewed, and costs milliseconds instead of
    repeating a multi-second trace. The SVG arrives from the browser, so it is
    validated as untrusted input - see ``svg_export._reject_unsafe``.
    """
    raw = await file.read()
    try:
        svg = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidSvgError("The SVG was not valid UTF-8 text.") from exc

    pdf = svg_to_pdf(svg)
    return Response(content=pdf, media_type="application/pdf")


if __name__ == "__main__":
    log.info(
        "Starting vectorization engine on http://%s:%s", settings.host, settings.port
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,  # our middleware already logs
    )
