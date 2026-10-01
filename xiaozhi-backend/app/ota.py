import time
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request

from .config import SUPPORTED_PROTOCOL_VERSION, get_settings


router = APIRouter(tags=["ota"])


def _validated_public_ws_url(url: str) -> str:
    parsed = urlparse(url)
    if not url or parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise HTTPException(
            status_code=503,
            detail="PUBLIC_WS_URL must be configured with a ws:// or wss:// URL reachable by ESP32",
        )
    return url


@router.api_route("/xiaozhi/ota/", methods=["GET", "POST"])
@router.api_route("/xiaozhi/ota", methods=["GET", "POST"], include_in_schema=False)
async def ota_config(request: Request) -> dict[str, Any]:
    settings = get_settings()
    return {
        "websocket": {
            "url": _validated_public_ws_url(settings.public_ws_url),
            "token": settings.device_token,
            "version": SUPPORTED_PROTOCOL_VERSION,
        },
        "server_time": {
            "timestamp": int(time.time() * 1000),
            "timezone_offset": settings.timezone_offset_minutes,
        },
        "device": {
            "device_id": request.headers.get("Device-Id", ""),
            "client_id": request.headers.get("Client-Id", ""),
        },
    }
