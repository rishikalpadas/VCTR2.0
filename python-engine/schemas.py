"""API response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "vectorize-engine"
    version: str
    engines: list[dict[str, Any]] = Field(default_factory=list)


class VectorizeResponse(BaseModel):
    success: bool = True
    svg: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody


class PresetListResponse(BaseModel):
    presets: list[dict[str, Any]]


class EngineListResponse(BaseModel):
    engines: list[dict[str, Any]]
