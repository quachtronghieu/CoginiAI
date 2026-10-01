import os

import pytest

from app.config import get_settings


# Unit tests use synthetic binary frames; real VAD is covered separately.
os.environ["VAD_ENABLED"] = "false"
os.environ["VOICE_DIAGNOSTICS_ENABLED"] = "false"
get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PUBLIC_WS_URL", "ws://172.20.10.2:8000/xiaozhi/v1/")
    monkeypatch.setenv("XIAOZHI_SERVER_SAMPLE_RATE", "24000")
    monkeypatch.setenv("XIAOZHI_FRAME_DURATION_MS", "60")
    monkeypatch.setenv("MAX_AUDIO_SECONDS", "30")
    monkeypatch.setenv("MAX_AUDIO_BYTES", "2097152")
    monkeypatch.setenv("AUTO_STOP_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("RNNOISE_ENABLED", "false")
    monkeypatch.setenv("VOICE_DIAGNOSTICS_ENABLED", "false")
    monkeypatch.setenv("VERIFIED_MENU_PATH", "")
    monkeypatch.delenv("XIAOZHI_DEVICE_TOKEN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
