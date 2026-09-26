"""ASGI application for the Laya Jev-compatible API."""

from __future__ import annotations

import os
import secrets
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field


class SystemOneRequest(BaseModel):
    """Subset of the Jev System One request, preserving unknown Jev fields."""

    model_config = ConfigDict(extra="allow")

    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1)
    model: str | None = None
    lang: str | None = None


@lru_cache
def get_router() -> Any:
    """Build the Laya router only when an inference request requires it."""
    cache_dir = os.getenv("LAYA_CACHE_DIR")
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir

    from laya import Router

    preload = os.getenv("LAYA_PRELOAD", "0").lower() in {"1", "true", "yes"}
    max_loaded = int(os.getenv("LAYA_MAX_LOADED", "1"))
    return Router(
        device=os.getenv("LAYA_DEVICE", "cpu"),
        max_loaded=max_loaded,
        preload=preload,
    )


def require_api_key(request: Request) -> None:
    """Protect inference only when API_KEY or LAYA_API_KEY is configured."""
    expected_key = os.getenv("API_KEY") or os.getenv("LAYA_API_KEY")
    if not expected_key:
        return

    authorization = request.headers.get("Authorization", "")
    scheme, _, supplied_key = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(supplied_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid Bearer token is required.",
        )


app = FastAPI(
    title="LayaJev",
    version="0.1.0",
    description="CPU-friendly, ARM64-ready Jev-compatible API powered by Laya.",
)
STATIC_DIR = Path(__file__).parent / "static"


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Any:
    """Expose server processing time without changing the Jev response body."""
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-MS"] = f"{(time.perf_counter() - start) * 1_000:.1f}"
    return response


@app.get("/", include_in_schema=False)
def test_console() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz", tags=["operations"])
def health_check() -> dict[str, str]:
    """Report process readiness without loading model weights."""
    return {"status": "ok"}


@app.post("/v1/systemone", dependencies=[Depends(require_api_key)], tags=["jev"])
def system_one(payload: SystemOneRequest) -> dict[str, Any]:
    """Run a typed decision using Laya's Jev-compatible request contract."""
    options: dict[str, Any] = {}
    if payload.model is not None:
        options["model"] = payload.model
    if payload.lang is not None:
        options["lang"] = payload.lang
    return get_router().predict(payload.state, payload.questions, **options)


def run() -> None:
    """Run the development or container server."""
    import uvicorn

    uvicorn.run("layajev.main:app", host="0.0.0.0", port=8000)