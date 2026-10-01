import json
from typing import Any

from .config import SUPPORTED_PROTOCOL_VERSION


class ProtocolError(ValueError):
    pass


class UnsupportedProtocolError(ProtocolError):
    pass


def parse_text_message(raw: str) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("Invalid JSON message") from exc

    if not isinstance(message, dict):
        raise ProtocolError("Message must be a JSON object")
    if not isinstance(message.get("type"), str):
        raise ProtocolError("Message type is required")
    return message


def validate_client_hello(message: dict[str, Any]) -> None:
    if message.get("type") != "hello":
        raise ProtocolError("First message must be hello")
    if message.get("transport") != "websocket":
        raise ProtocolError("Only websocket transport is supported")
    version = message.get("version")
    if version != SUPPORTED_PROTOCOL_VERSION:
        raise UnsupportedProtocolError(
            f"Unsupported protocol version: {version}. Only version {SUPPORTED_PROTOCOL_VERSION} is supported"
        )
    audio_params = message.get("audio_params")
    if not isinstance(audio_params, dict) or audio_params.get("format") != "opus":
        raise ProtocolError("Only Opus audio is supported")


def server_hello(
    session_id: str,
    sample_rate: int,
    frame_duration_ms: int,
) -> dict[str, Any]:
    return {
        "type": "hello",
        "transport": "websocket",
        "session_id": session_id,
        "audio_params": {
            "format": "opus",
            "sample_rate": sample_rate,
            "channels": 1,
            "frame_duration": frame_duration_ms,
        },
    }


def alert_message(message: str, session_id: str = "") -> dict[str, str]:
    alert = {
        "type": "alert",
        "status": "Backend",
        "message": message,
        "emotion": "neutral",
    }
    if session_id:
        alert["session_id"] = session_id
    return alert
