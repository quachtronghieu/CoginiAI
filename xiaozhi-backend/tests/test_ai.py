import asyncio
import json
import wave
from dataclasses import replace

import httpx
import numpy as np
import pytest

from app.ai import AiPipelineError, AsrResult, LocalAiPipeline, PcmAudio
from app.asr import FasterWhisperAsrService
from app.config import get_settings
from app.llm import OllamaLlmService
from app.tts import PiperTtsService


class FakeCodec:
    def __init__(self) -> None:
        self.encode_calls = 0

    async def decode_opus(self, frames: list[bytes]) -> PcmAudio:
        assert frames
        return PcmAudio(b"\x00\x00" * 160, 16_000)

    async def encode_opus(self, audio: PcmAudio) -> list[bytes]:
        self.encode_calls += 1
        return [b"opus"]

    def available(self) -> bool:
        return True


class EmptyAsr:
    async def transcribe(self, audio: PcmAudio) -> AsrResult:
        del audio
        return AsrResult("", "vi", 0.99)

    def available(self) -> bool:
        return True


class TrackingLlm:
    def __init__(self) -> None:
        self.calls = 0
        self.languages: list[str] = []
        self.language_probabilities: list[float] = []

    async def generate(
        self,
        transcript: str,
        language: str,
        language_probability: float,
        confidence: float = 0.0,
    ) -> str:
        del transcript, confidence
        self.calls += 1
        self.languages.append(language)
        self.language_probabilities.append(language_probability)
        return "answer"

    def available(self) -> bool:
        return True


class FakeTts:
    def __init__(self) -> None:
        self.languages: list[str] = []
        self.language_probabilities: list[float] = []

    async def synthesize(
        self,
        text: str,
        language: str,
        language_probability: float,
    ) -> PcmAudio:
        assert text
        self.languages.append(language)
        self.language_probabilities.append(language_probability)
        return PcmAudio(b"\x00\x00" * 240, 24_000)

    def available(self) -> bool:
        return True

    def health(self) -> dict[str, bool]:
        return {"tts": True, "tts_vi": True, "tts_en": True}


class ResultAsr:
    def __init__(self, language: str) -> None:
        self.language = language

    async def transcribe(self, audio: PcmAudio) -> AsrResult:
        del audio
        return AsrResult("hello", self.language, 0.95)

    def available(self) -> bool:
        return True


def test_empty_asr_does_not_call_llm(caplog) -> None:
    llm = TrackingLlm()
    pipeline = LocalAiPipeline(FakeCodec(), EmptyAsr(), llm, FakeTts())

    caplog.set_level("WARNING")
    with pytest.raises(AiPipelineError, match="empty transcript"):
        asyncio.run(pipeline.run([b"opus"]))

    assert llm.calls == 0
    assert "ASR empty transcript ignored" in caplog.text


def test_faster_whisper_adapter_contract() -> None:
    settings = replace(get_settings(), asr_model="local-model", asr_language="vi")
    service = FasterWhisperAsrService(settings)

    class Segment:
        text = " xin chao "

    class Model:
        def transcribe(self, samples, **kwargs):
            assert samples.dtype == np.float32
            assert kwargs["language"] == "vi"
            class Info:
                language = "vi"
                language_probability = 0.98

            return iter([Segment()]), Info()

    service._model = Model()
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 3200, 16_000)))

    assert result == AsrResult("xin chao", "vi", 0.98)


def test_blank_asr_language_enables_auto_detection() -> None:
    settings = replace(get_settings(), asr_model="local-model", asr_language="")
    service = FasterWhisperAsrService(settings)

    class Model:
        def transcribe(self, samples, **kwargs):
            del samples
            assert kwargs["language"] is None

            class Segment:
                text = " hello "

            class Info:
                language = "en"
                language_probability = 0.97

            return iter([Segment()]), Info()

    service._model = Model()
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 3200, 16_000)))

    assert result.text == "hello"
    assert result.language == "en"
    assert result.language_probability == 0.97


def test_short_english_non_vi_en_auto_result_retries_english() -> None:
    settings = replace(get_settings(), asr_model="local-model", asr_language="")
    service = FasterWhisperAsrService(settings)

    class Segment:
        def __init__(self, text: str, avg_logprob: float, no_speech_prob: float = 0.0):
            self.text = text
            self.avg_logprob = avg_logprob
            self.no_speech_prob = no_speech_prob

    class Model:
        def transcribe(self, samples, **kwargs):
            del samples
            language = kwargs["language"]
            if language is None:
                return iter([Segment("Good morning", -0.9)]), type(
                    "Info", (), {"language": "ru", "language_probability": 0.94}
                )()
            if language == "en":
                return iter([Segment("Good morning", -0.1)]), type(
                    "Info", (), {"language": "en", "language_probability": 0.82}
                )()
            return iter([Segment("Chao buoi sang", -0.8)]), type(
                "Info", (), {"language": "vi", "language_probability": 0.65}
            )()

    service._model = Model()
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 3200, 16_000)))

    assert result == AsrResult("Good morning", "en", 0.82, -0.1)


def test_short_vietnamese_low_confidence_retries_vietnamese() -> None:
    settings = replace(get_settings(), asr_model="local-model", asr_language="")
    service = FasterWhisperAsrService(settings)

    class Segment:
        def __init__(self, text: str, avg_logprob: float):
            self.text = text
            self.avg_logprob = avg_logprob
            self.no_speech_prob = 0.0

    class Model:
        def transcribe(self, samples, **kwargs):
            del samples
            language = kwargs["language"]
            if language is None:
                return iter([Segment("Xin chao", -0.5)]), type(
                    "Info", (), {"language": "vi", "language_probability": 0.55}
                )()
            if language == "vi":
                return iter([Segment("Xin chào", -0.1)]), type(
                    "Info", (), {"language": "vi", "language_probability": 0.86}
                )()
            return iter([Segment("Sin chow", -0.8)]), type(
                "Info", (), {"language": "en", "language_probability": 0.60}
            )()

    service._model = Model()
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 3200, 16_000)))

    assert result == AsrResult("Xin chào", "vi", 0.86, -0.1)


def test_cyrillic_auto_result_falls_back_to_valid_english() -> None:
    settings = replace(get_settings(), asr_model="local-model", asr_language="")
    service = FasterWhisperAsrService(settings)

    class Segment:
        def __init__(self, text: str, avg_logprob: float):
            self.text = text
            self.avg_logprob = avg_logprob
            self.no_speech_prob = 0.0

    class Model:
        def transcribe(self, samples, **kwargs):
            del samples
            language = kwargs["language"]
            if language is None:
                return iter([Segment("О, и утро!", -0.05)]), type(
                    "Info", (), {"language": "ru", "language_probability": 0.99}
                )()
            if language == "en":
                return iter([Segment("Oh, good morning!", -0.1)]), type(
                    "Info", (), {"language": "en", "language_probability": 0.90}
                )()
            return iter([Segment("Ồ, chào buổi sáng!", -0.7)]), type(
                "Info", (), {"language": "vi", "language_probability": 0.85}
            )()

    service._model = Model()
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 3200, 16_000)))

    assert result.language == "en"
    assert result.text == "Oh, good morning!"


@pytest.mark.parametrize("language", ["vi", "en"])
def test_language_propagates_through_pipeline(language: str) -> None:
    llm = TrackingLlm()
    tts = FakeTts()
    pipeline = LocalAiPipeline(FakeCodec(), ResultAsr(language), llm, tts)

    result = asyncio.run(pipeline.run([b"opus"]))

    assert result.language == language
    assert llm.languages == [language]
    assert llm.language_probabilities == [0.95]
    assert tts.languages == [language]
    assert tts.language_probabilities == [0.95]


def test_ollama_adapter_uses_local_generate_api() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(request.content)
        assert payload["messages"][-1] == {
            "role": "user",
            "content": "Please answer in English",
        }
        assert payload["think"] is False
        system = payload["messages"][0]["content"]
        assert "ASR secondary hint: vi with confidence 0.95" in system
        assert "transcript text has priority" in system
        assert "Ignore the hint when it conflicts" in system
        assert "explicit user request" in system
        assert "1 to 3 concise spoken sentences" in system
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello!"}},
        )

    settings = replace(
        get_settings(),
        llm_base_url="http://127.0.0.1:11434",
        llm_model="qwen3:4b",
    )
    service = OllamaLlmService(settings, httpx.MockTransport(handler))

    assert asyncio.run(service.generate("Please answer in English", "vi", 0.95)) == "Hello!"


def test_ollama_ignores_low_confidence_asr_language_hint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][-1]["content"] == "What is pho?"
        system = payload["messages"][0]["content"]
        assert "No reliable ASR language hint is available" in system
        assert "ASR secondary hint: vi" not in system
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello!"}},
        )

    settings = replace(
        get_settings(),
        llm_base_url="http://127.0.0.1:11434",
        llm_model="qwen3:4b",
        language_hint_min_probability=0.80,
    )
    service = OllamaLlmService(settings, httpx.MockTransport(handler))

    assert asyncio.run(service.generate("What is pho?", "vi", 0.79)) == "Hello!"


def test_ollama_strips_qwen_thinking_before_tts() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<think>internal reasoning</think>Final answer.",
                },
                "done_reason": "stop",
            },
        )

    service = OllamaLlmService(
        replace(get_settings(), llm_base_url="http://127.0.0.1:11434"),
        httpx.MockTransport(handler),
    )

    assert asyncio.run(service.generate("What is pho?", "en", 0.99)) == "Final answer."


def test_ollama_rejects_truncated_thinking_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "<think>incomplete reasoning",
                },
                "done_reason": "length",
            },
        )

    service = OllamaLlmService(
        replace(get_settings(), llm_base_url="http://127.0.0.1:11434"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(AiPipelineError, match="truncated"):
        asyncio.run(service.generate("What is pho?", "en", 0.99))


def test_ollama_readiness_requires_configured_model() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://127.0.0.1:11434/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:4b"}]},
        )

    settings = replace(
        get_settings(),
        llm_base_url="http://127.0.0.1:11434",
        llm_model="qwen3:4b",
    )
    service = OllamaLlmService(settings, httpx.MockTransport(handler))

    assert asyncio.run(service.ready()) is True


def test_ollama_readiness_is_false_when_model_is_missing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"models": []})

    settings = replace(
        get_settings(),
        llm_base_url="http://127.0.0.1:11434",
        llm_model="qwen3:4b",
    )
    service = OllamaLlmService(settings, httpx.MockTransport(handler))

    assert asyncio.run(service.ready()) is False


def test_piper_adapter_returns_pcm(tmp_path) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model-placeholder")
    (tmp_path / "voice.onnx.json").write_text("{}")
    settings = replace(
        get_settings(),
        tts_model=str(model_path),
        tts_model_vi="",
        tts_model_en="",
    )
    service = PiperTtsService(settings)

    class Voice:
        def synthesize_wav(self, text: str, wav_file) -> None:
            assert text == "xin chao"
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22_050)
            wav_file.writeframes(b"\x00\x00" * 100)

    service._voices["legacy"] = Voice()
    audio = asyncio.run(service.synthesize("xin chao", "vi", 0.95))

    assert audio.sample_rate == 22_050
    assert audio.channels == 1
    assert len(audio.data) == 200


def test_dual_piper_routing_and_independent_cache(tmp_path, monkeypatch) -> None:
    vi_model = tmp_path / "vi.onnx"
    en_model = tmp_path / "en.onnx"
    vi_model.write_bytes(b"vi")
    en_model.write_bytes(b"en")
    (tmp_path / "vi.onnx.json").write_text("{}")
    (tmp_path / "en.onnx.json").write_text("{}")
    settings = replace(
        get_settings(),
        tts_model="",
        tts_model_vi=str(vi_model),
        tts_model_en=str(en_model),
        tts_default_language="vi",
    )
    service = PiperTtsService(settings)
    loaded: list[str] = []

    def load_voice(model_path: str, config_path: str):
        del config_path
        loaded.append(model_path)
        return object()

    monkeypatch.setattr(service, "_load_voice", load_voice)

    vi_voice = service._get_voice("vi")
    en_voice = service._get_voice("en")
    assert service._get_voice("vi") is vi_voice
    assert service._get_voice("en") is en_voice
    assert loaded == [str(vi_model), str(en_model)]
    assert service.health() == {"tts": True, "tts_vi": True, "tts_en": True}


def test_tts_routes_from_answer_language_and_safe_fallback(tmp_path) -> None:
    vi_model = tmp_path / "vi.onnx"
    en_model = tmp_path / "en.onnx"
    vi_model.write_bytes(b"vi")
    en_model.write_bytes(b"en")
    (tmp_path / "vi.onnx.json").write_text("{}")
    (tmp_path / "en.onnx.json").write_text("{}")
    service = PiperTtsService(
        replace(
            get_settings(),
            tts_model="",
            tts_model_vi=str(vi_model),
            tts_model_en=str(en_model),
            tts_default_language="vi",
        )
    )

    assert service.select_language("Xin chào bạn", "en") == "vi"
    assert service.select_language("Hello, how are you?", "vi") == "en"
    assert service.select_language("Hi", "vi", 0.99) == "en"
    assert service.select_language("OK", "vi", 0.99) == "en"
    assert service.select_language("Certainly.", "vi", 0.99) == "en"
    assert service.select_language("toi co the giup ban", "en", 0.99) == "vi"
    assert service.select_language("12345", "fr") == "vi"
    assert service.select_language("12345", "en", 0.79) == "vi"

def test_edge_tts_falls_back_to_piper_on_error(tmp_path, monkeypatch):
    from app.tts import EdgeTtsService

    vi_model = tmp_path / "vi.onnx"
    vi_model.write_bytes(b"vi")
    (tmp_path / "vi.onnx.json").write_text("{}")
    settings = replace(
        get_settings(),
        tts_model_vi=str(vi_model),
        tts_default_language="vi",
    )
    service = EdgeTtsService(settings)

    class MockPiper:
        def available(self):
            return True
        async def synthesize(self, text, language, language_probability):
            return PcmAudio(b"\x00\x00" * 120, 24000)

    service._piper_fallback = MockPiper()

    # Simulate edge_tts raising NoAudioReceived
    def mock_stream():
        raise RuntimeError("No audio was received")

    class MockCommunicate:
        def __init__(self, *args, **kwargs):
            pass
        async def stream(self):
            raise RuntimeError("No audio was received")
            yield {}

    import edge_tts
    monkeypatch.setattr(edge_tts, "Communicate", MockCommunicate)

    result = asyncio.run(service.synthesize("Món ngon Cà Mau", "vi"))
    assert result.sample_rate == 24000
    assert len(result.data) == 240


def test_clean_speech_text_normalizes_currency_and_pauses() -> None:
    from app.tts import clean_speech_text
    raw = "Lẩu gà lá é; giá 200.000 VNĐ. Bánh ướt: giá 40.000 VND (ghi chú)."
    cleaned = clean_speech_text(raw)
    assert "200 nghìn đồng" in cleaned
    assert "40 nghìn đồng" in cleaned
    assert ";" not in cleaned
    assert ":" not in cleaned
    assert "(" not in cleaned and ")" not in cleaned


def test_piper_synthesizes_multiple_sentences_with_silence() -> None:
    from app.tts import PiperTtsService
    from collections import namedtuple
    settings = get_settings()
    service = PiperTtsService(settings)

    AudioChunk = namedtuple("AudioChunk", ["audio_int16_bytes"])
    Config = namedtuple("Config", ["sample_rate"])

    class StreamingVoice:
        config = Config(sample_rate=22050)
        def synthesize(self, text, syn_config=None):
            return [AudioChunk(b"\x01\x01" * 100), AudioChunk(b"\x02\x02" * 100)]

    service._voices["legacy"] = StreamingVoice()
    audio = asyncio.run(service.synthesize("Câu một. Câu hai.", "vi", 0.95))
    assert audio.sample_rate == 22050
    # Expected: 200 samples chunk 1 + silence samples (22050 * 0.25 = 5512 samples) + 200 samples chunk 2
    assert len(audio.data) > 400

