from fastapi import FastAPI
from contextlib import asynccontextmanager

from .config import get_settings
from .diagnostics import configure_diagnostics_logging


settings = get_settings()
configure_diagnostics_logging(settings.diagnostics_enabled)

# Build audio services only after the local diagnostics handler is active so
# RNNoise's runtime-contract audit is retained in voice-session.log.
from .ota import router as ota_router
from . import websocket as websocket_module


@asynccontextmanager
async def lifespan(app):
    tts = getattr(websocket_module.ai_pipeline, "tts", None)
    warmup = getattr(tts, "warmup", None)
    if settings.tts_warmup_enabled and warmup is not None:
        await warmup()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(ota_router)
app.include_router(websocket_module.router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ai", tags=["system"])
async def ai_health() -> dict[str, bool]:
    status = await websocket_module.ai_readiness()
    status["vad"] = websocket_module.vad_factory.available()
    return status
