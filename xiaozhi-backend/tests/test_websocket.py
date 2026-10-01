import json
import asyncio
from contextlib import contextmanager
from threading import Event
from time import monotonic

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.websocket as websocket_module
from app.ai import (
    AiUnavailableError,
    AiResponse,
    AsrResult,
    EmptyUtteranceError,
    LocalAiPipeline,
    PcmAudio,
)
from app.codec import PyAvOpusCodec
from app.config import get_settings
from app.main import app
from app.session import AudioSession


client = TestClient(app)


def hello(version: int = 1) -> dict:
    return {
        "type": "hello",
        "version": version,
        "transport": "websocket",
        "features": {"mcp": True},
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60,
        },
    }


class FakeAiPipeline:
    def __init__(self) -> None:
        self.calls: list[list[bytes]] = []

    async def run(self, opus_frames: list[bytes]) -> AiResponse:
        self.calls.append(list(opus_frames))
        return AiResponse(
            transcript="hello",
            answer="world",
            audio_frames=[b"tts-opus"],
        )

    def health(self) -> dict[str, bool]:
        return {"codec": True, "asr": True, "llm": True, "tts": True}


class BlockingAiPipeline:
    def __init__(self) -> None:
        self.started = Event()
        self.cancelled = Event()

    async def run(self, opus_frames: list[bytes]) -> AiResponse:
        del opus_frames
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def health(self) -> dict[str, bool]:
        return {"codec": True, "asr": True, "llm": True, "tts": True}


class RecordingWebSocket:
    def __init__(self) -> None:
        self.events: list[tuple[float, object]] = []
        self.first_frame = asyncio.Event()
        self.close_code: int | None = None
        self.close_reason = ""

    async def send_json(self, message: object) -> None:
        self.events.append((monotonic(), message))

    async def send_bytes(self, frame: bytes) -> None:
        self.events.append((monotonic(), frame))
        self.first_frame.set()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.close_reason = reason


@contextmanager
def open_hello():
    with client.websocket_connect(
        "/xiaozhi/v1/",
        headers={"Device-Id": "aa:bb:cc:dd:ee:ff", "Client-Id": "test-client"},
    ) as websocket:
        websocket.send_json(hello())
        assert websocket.receive_json()["type"] == "hello"
        yield websocket


def test_websocket_hello_protocol_v1() -> None:
    with open_hello() as websocket:
        websocket.send_json({"type": "abort"})


def test_websocket_rejects_unsupported_protocol() -> None:
    with client.websocket_connect("/xiaozhi/v1/") as websocket:
        websocket.send_json(hello(version=2))
        alert = websocket.receive_json()
        assert "Unsupported protocol version" in alert["message"]
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()


def test_valid_token(monkeypatch) -> None:
    monkeypatch.setenv("XIAOZHI_DEVICE_TOKEN", "secret")
    get_settings.cache_clear()
    with client.websocket_connect(
        "/xiaozhi/v1/", headers={"Authorization": "Bearer secret"}
    ) as websocket:
        websocket.send_json(hello())
        assert websocket.receive_json()["type"] == "hello"


def test_invalid_token(monkeypatch) -> None:
    monkeypatch.setenv("XIAOZHI_DEVICE_TOKEN", "secret")
    get_settings.cache_clear()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/xiaozhi/v1/", headers={"Authorization": "Bearer wrong"}
        ):
            pass


def test_invalid_json_does_not_kill_server() -> None:
    with client.websocket_connect("/xiaozhi/v1/") as websocket:
        websocket.send_text("not-json")
        assert websocket.receive_json()["type"] == "alert"
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert client.get("/health").json() == {"status": "ok"}


def test_binary_audio_and_event_order(monkeypatch) -> None:
    fake = FakeAiPipeline()
    monkeypatch.setattr(websocket_module, "ai_pipeline", fake)

    with open_hello() as websocket:
        websocket.send_json({"type": "listen", "state": "start", "mode": "manual"})
        websocket.send_bytes(b"input-opus")
        websocket.send_json({"type": "listen", "state": "stop"})

        messages = []
        for _ in range(6):
            incoming = websocket.receive()
            if incoming.get("text") is not None:
                messages.append(json.loads(incoming["text"]))
            else:
                messages.append(incoming["bytes"])

    assert fake.calls == [[b"input-opus"]]
    assert [m["type"] if isinstance(m, dict) else "binary" for m in messages] == [
        "stt",
        "llm",
        "tts",
        "tts",
        "binary",
        "tts",
    ]


def test_tts_stop_waits_for_expected_playback_duration(caplog) -> None:
    websocket = RecordingWebSocket()
    result = AiResponse(
        transcript="hello",
        answer="world",
        audio_frames=[b"one", b"two", b"three"],
    )

    caplog.set_level("INFO")
    asyncio.run(websocket_module._send_ai_result(websocket, "session", result, 60))

    frame_times = [
        timestamp
        for timestamp, event in websocket.events
        if isinstance(event, bytes)
    ]
    stop_time = next(
        timestamp
        for timestamp, event in websocket.events
        if isinstance(event, dict) and event.get("state") == "stop"
    )
    assert len(frame_times) == 3
    assert stop_time - frame_times[0] >= 0.17
    assert "frames=3" in caplog.text
    assert "expected_playback_ms=180.0" in caplog.text


def test_abort_cancels_tts_pacing_immediately() -> None:
    websocket = RecordingWebSocket()

    async def blocked_sleep(seconds: float) -> None:
        del seconds
        await asyncio.Future()

    async def scenario() -> float:
        task = asyncio.create_task(
            websocket_module._send_ai_result(
                websocket,
                "session",
                AiResponse(
                    transcript="hello",
                    answer="world",
                    audio_frames=[b"one", b"two", b"three"],
                ),
                60,
            )
        )
        await websocket.first_frame.wait()
        started = monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return monotonic() - started

    original_sleep = websocket_module.asyncio.sleep
    websocket_module.asyncio.sleep = blocked_sleep
    try:
        elapsed = asyncio.run(scenario())
    finally:
        websocket_module.asyncio.sleep = original_sleep

    assert elapsed < 0.2
    assert any(
        isinstance(event, dict) and event.get("state") == "stop"
        for _, event in websocket.events
    )


def test_empty_utterance_is_ignored_without_alert(monkeypatch) -> None:
    class EmptyPipeline:
        async def run(self, frames: list[bytes]) -> AiResponse:
            del frames
            raise EmptyUtteranceError("ASR returned an empty transcript")

    websocket = RecordingWebSocket()
    monkeypatch.setattr(websocket_module, "ai_pipeline", EmptyPipeline())

    asyncio.run(
        websocket_module._process_utterance(
            websocket,
            "session",
            [b"noise"],
            websocket_module.FinalizeReason.MANUAL_STOP,
        )
    )

    assert websocket.events == []


def test_empty_utterance_ends_without_reply_or_automatic_rearm(monkeypatch) -> None:
    class EmptyPipeline:
        async def run(self, frames: list[bytes]) -> AiResponse:
            del frames
            raise EmptyUtteranceError("ASR returned an empty transcript")

    websocket = RecordingWebSocket()
    rearmed = []
    monkeypatch.setattr(websocket_module, "ai_pipeline", EmptyPipeline())

    asyncio.run(
        websocket_module._process_utterance(
            websocket,
            "session",
            [b"noise"],
            websocket_module.FinalizeReason.AUTO_TIMEOUT,
            resume_after_empty=lambda: rearmed.append(True),
        )
    )

    assert rearmed == []
    assert websocket.events == []


def test_weak_vad_is_passed_to_combined_gate_without_reply(monkeypatch) -> None:
    class GatedPipeline:
        def __init__(self):
            self.calls = []

        async def run(self, frames, **kwargs):
            self.calls.append((list(frames), kwargs["vad_evidence"]))
            from app.ai import NonSpeechUtteranceError
            raise NonSpeechUtteranceError("weak VAD and ASR")

        async def record_rejected_utterance(self, frames, reason):
            pytest.fail("Weak VAD was hard-rejected before ASR evidence")

    class SilentSession:
        session_id = "silent-session"
        elapsed_seconds = 4.0
        last_utterance_had_speech = True
        last_vad_evidence = {"available": True, "max_contiguous_speech_ms": 160.0}
        persona = "default"
        history = []
        customer_profile = None

        def update_persona(self, transcript):
            del transcript
            return False

        def add_history(self, transcript, answer):
            del transcript, answer

        def take_audio(self):
            self.last_utterance_had_speech = False
            return [b"noise"]

    pipeline = GatedPipeline()
    websocket = RecordingWebSocket()
    monkeypatch.setattr(websocket_module, "ai_pipeline", pipeline)

    async def exercise():
        task = await websocket_module._start_inference(
            websocket,
            SilentSession(),
            websocket_module.FinalizeReason.AUTO_TIMEOUT,
            None,
        )
        assert task is not None
        await task

    asyncio.run(exercise())

    assert pipeline.calls == [([b"noise"], {"available": True, "max_contiguous_speech_ms": 160.0})]
    assert websocket.events == []


@pytest.mark.parametrize("speech", [False, True, None])
def test_explicit_no_speech_blocks_before_pipeline(monkeypatch, speech):
    class Session:
        session_id = "gate-test"
        elapsed_seconds = 3.3
        last_vad_evidence = {"available": speech is not None, "speech_detected": speech}
        def take_audio(self):
            return [b"pcm"]
    calls = []
    async def record(*args):
        calls.append("record")
    async def process(*args, **kwargs):
        calls.append("asr")
    monkeypatch.setattr(websocket_module, "_record_vad_rejection", record)
    monkeypatch.setattr(websocket_module, "_process_utterance", process)
    socket = RecordingWebSocket()
    async def exercise():
        task = await websocket_module._start_inference(socket, Session(), websocket_module.FinalizeReason.MANUAL_STOP, None)
        await task
    asyncio.run(exercise())
    assert calls == (["record"] if speech is False else ["asr"])
    assert socket.close_code == (1000 if speech is False else None)


def test_audio_buffer_limit_finalizes(monkeypatch) -> None:
    monkeypatch.setenv("MAX_AUDIO_BYTES", "3")
    monkeypatch.setenv("AUTO_STOP_TIMEOUT_SECONDS", "100")
    get_settings.cache_clear()
    fake = FakeAiPipeline()
    monkeypatch.setattr(websocket_module, "ai_pipeline", fake)

    with open_hello() as websocket:
        websocket.send_json({"type": "listen", "state": "start", "mode": "auto"})
        websocket.send_bytes(b"abc")
        websocket.send_bytes(b"d")
        for _ in range(6):
            websocket.receive()

    assert fake.calls == [[b"abc"]]


def test_ai_unavailable_closes_conversation_without_backend_alert(monkeypatch) -> None:
    class UnavailablePipeline:
        async def run(self, frames: list[bytes]) -> AiResponse:
            del frames
            raise AiUnavailableError("Local LLM timed out")

    monkeypatch.setattr(websocket_module, "ai_pipeline", UnavailablePipeline())

    with open_hello() as websocket:
        websocket.send_json({"type": "listen", "state": "start", "mode": "manual"})
        websocket.send_bytes(b"input-opus")
        websocket.send_json({"type": "listen", "state": "stop"})
        with pytest.raises(WebSocketDisconnect) as exc_info:
            websocket.receive_json()

        assert exc_info.value.code == 1000


def test_two_sequential_utterances_receive_two_answers(monkeypatch) -> None:
    fake = FakeAiPipeline()
    monkeypatch.setattr(websocket_module, "ai_pipeline", fake)

    with open_hello() as websocket:
        for index in range(2):
            websocket.send_json(
                {"type": "listen", "state": "start", "mode": "manual"}
            )
            websocket.send_bytes(f"input-opus-{index}".encode())
            websocket.send_json({"type": "listen", "state": "stop"})
            messages = [websocket.receive() for _ in range(6)]

            assert [
                "binary" if message.get("bytes") is not None else json.loads(message["text"])["type"]
                for message in messages
            ] == ["stt", "llm", "tts", "tts", "binary", "tts"]

    assert fake.calls == [[b"input-opus-0"], [b"input-opus-1"]]


def test_ai_pipeline_timeout_closes_conversation_without_alert(monkeypatch) -> None:
    class SlowPipeline:
        async def run(self, frames: list[bytes]) -> AiResponse:
            del frames
            await asyncio.sleep(1)
            return AiResponse("", "", [])

    settings = type("Settings", (), {"ai_pipeline_timeout_seconds": 0.01})()
    monkeypatch.setattr(websocket_module, "ai_pipeline", SlowPipeline())
    monkeypatch.setattr(websocket_module, "get_settings", lambda: settings)

    websocket = RecordingWebSocket()
    asyncio.run(
        websocket_module._process_utterance(
            websocket,
            "session",
            [b"input-opus"],
            websocket_module.FinalizeReason.MANUAL_STOP,
        )
    )

    assert websocket.events == []
    assert websocket.close_code == 1000
    assert websocket.close_reason == "AI turn ended"


def test_invalid_opus_frame_returns_alert_without_crashing(monkeypatch) -> None:
    class PassAsr:
        async def transcribe(self, audio: PcmAudio) -> AsrResult:
            del audio
            return AsrResult("hello", "en", 0.9)

        def available(self) -> bool:
            return True

    class PassLlm:
        async def generate(
            self, text: str, language: str, language_probability: float
        ) -> str:
            assert language == "en"
            assert language_probability == 0.9
            return text

        def available(self) -> bool:
            return True

    class PassTts:
        async def synthesize(
            self, text: str, language: str, language_probability: float
        ) -> PcmAudio:
            assert language == "en"
            assert language_probability == 0.9
            return PcmAudio(b"\x00\x00" * 240, 24_000)

        def available(self) -> bool:
            return True

        def health(self) -> dict[str, bool]:
            return {"tts": True, "tts_vi": True, "tts_en": True}

    monkeypatch.setattr(
        websocket_module,
        "ai_pipeline",
        LocalAiPipeline(PyAvOpusCodec(), PassAsr(), PassLlm(), PassTts()),
    )
    with open_hello() as websocket:
        websocket.send_json({"type": "listen", "state": "start", "mode": "manual"})
        websocket.send_bytes(b"not-an-opus-packet")
        websocket.send_json({"type": "listen", "state": "stop"})
        alert = websocket.receive_json()
        assert alert["type"] == "alert"
        websocket.send_json({"type": "abort"})


def test_abort_cancels_inference(monkeypatch) -> None:
    pipeline = BlockingAiPipeline()
    monkeypatch.setattr(websocket_module, "ai_pipeline", pipeline)

    with open_hello() as websocket:
        websocket.send_json({"type": "listen", "state": "start", "mode": "manual"})
        websocket.send_bytes(b"input-opus")
        websocket.send_json({"type": "listen", "state": "stop"})
        assert pipeline.started.wait(2)
        websocket.send_json({"type": "abort"})
        assert pipeline.cancelled.wait(2)


def test_disconnect_cancels_inference(monkeypatch) -> None:
    pipeline = BlockingAiPipeline()
    monkeypatch.setattr(websocket_module, "ai_pipeline", pipeline)

    with open_hello() as websocket:
        websocket.send_json({"type": "listen", "state": "start", "mode": "manual"})
        websocket.send_bytes(b"input-opus")
        websocket.send_json({"type": "listen", "state": "stop"})
        assert pipeline.started.wait(2)

    assert pipeline.cancelled.wait(2)


def test_disconnect_clears_session(monkeypatch) -> None:
    created: list[AudioSession] = []
    original = websocket_module.AudioSession

    class TrackingAudioSession(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(websocket_module, "AudioSession", TrackingAudioSession)
    with open_hello() as websocket:
        websocket.send_json({"type": "listen", "state": "start", "mode": "manual"})
        websocket.send_bytes(b"input-opus")

    assert created[0].frame_count == 0
    assert created[0].total_bytes == 0
    assert created[0].active is False
