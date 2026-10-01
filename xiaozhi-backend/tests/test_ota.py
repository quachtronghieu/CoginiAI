from fastapi.testclient import TestClient

from app.main import app
import app.websocket as websocket_module


client = TestClient(app)


def test_ota_returns_websocket_config() -> None:
    response = client.post(
        "/xiaozhi/ota/",
        headers={"Device-Id": "aa:bb:cc:dd:ee:ff", "Client-Id": "test-client"},
        json={"version": "test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["websocket"]["url"].endswith("/xiaozhi/v1/")
    assert data["websocket"]["version"] == 1
    assert data["device"]["device_id"] == "aa:bb:cc:dd:ee:ff"


def test_ota_rejects_missing_public_url(monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.delenv("PUBLIC_WS_URL")
    get_settings.cache_clear()
    response = client.get("/xiaozhi/ota/")

    assert response.status_code == 503
    assert "PUBLIC_WS_URL" in response.json()["detail"]


def test_ai_health_endpoint(monkeypatch) -> None:
    class Pipeline:
        def health(self) -> dict[str, bool]:
            return {
                "codec": True,
                "asr": True,
                "llm": True,
                "rnnoise": True,
                "tts": True,
                "tts_vi": True,
                "tts_en": True,
            }

    monkeypatch.setattr(websocket_module, "ai_pipeline", Pipeline())
    response = client.get("/health/ai")

    assert response.status_code == 200
    assert set(response.json()) == {
        "codec",
        "asr",
        "llm",
        "rnnoise",
        "tts",
        "tts_vi",
        "tts_en",
        "vad",
    }
