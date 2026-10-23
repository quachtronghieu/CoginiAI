import os
from dataclasses import replace

import pytest
from dotenv import dotenv_values

from app.ai import LocalAiPipeline, build_ai_pipeline
from app.config import PROJECT_ROOT, Settings, get_settings
from scripts.run_asr_trial import configure_trial_environment, parse_args


def test_daily_defaults_match_example(monkeypatch):
    keys = (
        "RNNOISE_ASR_MODE", "RNNOISE_ENABLED", "VAD_THRESHOLD",
        "VAD_MIN_SILENCE_MS", "AUTO_STOP_TIMEOUT_SECONDS",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    example = dotenv_values(PROJECT_ROOT / ".env.example")

    assert Settings.rnnoise_asr_mode == settings.rnnoise_asr_mode == example["RNNOISE_ASR_MODE"] == "raw"
    assert settings.rnnoise_enabled is False
    assert example["RNNOISE_ENABLED"] == "false"
    assert settings.vad_threshold == float(example["VAD_THRESHOLD"]) == .6
    assert settings.vad_min_silence_ms == int(example["VAD_MIN_SILENCE_MS"]) == 1000
    assert settings.auto_stop_timeout_seconds == float(example["AUTO_STOP_TIMEOUT_SECONDS"]) == 6
    assert example["ASR_MODEL"] == "small"


@pytest.mark.parametrize("threshold", ["-0.1", "1.1"])
def test_vad_threshold_must_be_probability(monkeypatch, threshold):
    monkeypatch.setenv("VAD_THRESHOLD", threshold)
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="VAD_THRESHOLD"):
        get_settings()


def test_raw_pipeline_does_not_require_rnnoise_runtime(monkeypatch):
    import app.noise as noise_module
    import app.tts as tts_module

    def forbidden_noise(*args, **kwargs):
        pytest.fail("Raw pipeline initialized RNNoise")

    class ReadyTts:
        def available(self):
            return True

        def health(self):
            return {"tts": True}

    monkeypatch.setattr(noise_module, "RnnoiseNoiseSuppressor", forbidden_noise)
    monkeypatch.setattr(tts_module, "create_tts_service", lambda settings: ReadyTts())
    settings = replace(
        get_settings(), rnnoise_enabled=True, rnnoise_asr_mode="raw",
        asr_model="small", llm_base_url="http://127.0.0.1:11434", llm_model="gemma3:4b",
    )
    pipeline = build_ai_pipeline(settings)
    assert isinstance(pipeline, LocalAiPipeline)
    assert pipeline.noise_suppressor is None


@pytest.mark.parametrize("model", ["small", "turbo"])
@pytest.mark.parametrize("mode", ["raw", "shadow"])
def test_trial_modes_and_rollback_are_process_local(monkeypatch, tmp_path, model, mode):
    keys = (
        "ASR_MODEL", "ASR_COMPUTE_TYPE", "RNNOISE_ASR_MODE", "RNNOISE_ENABLED",
        "VOICE_DIAGNOSTICS_ENABLED", "VOICE_DIAGNOSTICS_DIR", "VOICE_DIAGNOSTICS_SLOTS",
    )
    for key in keys:
        monkeypatch.setenv(key, "previous-value")
    env_file = tmp_path / ".env"
    env_file.write_text("ASR_MODEL=small\nRNNOISE_ASR_MODE=raw\n", encoding="utf-8")
    original = env_file.read_bytes()
    args = parse_args(["--model", model, "--rnnoise-mode", mode])
    configure_trial_environment(args, tmp_path, tmp_path / "trial")

    assert os.environ["ASR_MODEL"] == (str(tmp_path / "models/faster-whisper-large-v3-turbo") if model == "turbo" else "small")
    assert os.environ["ASR_COMPUTE_TYPE"] == ("int8_float16" if model == "turbo" else "float16")
    assert os.environ["RNNOISE_ASR_MODE"] == mode
    assert os.environ["RNNOISE_ENABLED"] == ("true" if mode == "shadow" else "false")
    assert env_file.read_bytes() == original
    assert parse_args(["--model", model]).rnnoise_mode == "raw"
