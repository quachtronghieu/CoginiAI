"""Opt-in, bounded local voice diagnostics. Never exposed by HTTP routes."""

from __future__ import annotations

import inspect
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
from datetime import datetime, timezone
import wave

from .config import PROJECT_ROOT


def supported_kwargs(callable_object, kwargs: dict) -> dict:
    """Adapt old test/service signatures without swallowing internal TypeErrors."""
    parameters = inspect.signature(callable_object).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in parameters}


class VoiceDiagnostics:
    def __init__(self, directory: Path | None = None, slots: int = 24) -> None:
        self.directory = directory or Path(os.getenv("VOICE_DIAGNOSTICS_DIR", str(PROJECT_ROOT / "logs" / "voice")))
        self.slots = slots
        self._counter = 0
        self._lock = threading.Lock()

    def _metadata_path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _write_audio(self, path: Path, audio) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(audio.channels)
            output.setsampwidth(2)
            output.setframerate(audio.sample_rate)
            output.writeframes(audio.data)

    def _write_metadata(self, key: str, metadata: dict) -> None:
        path = self._metadata_path(key)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _claim_benchmark_case(self) -> dict:
        state_path = self.directory / "benchmark-state.json"
        if not state_path.exists():
            return {}
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not state.get("active"):
            return {}
        index = int(state.get("next_index", 0))
        cases = state.get("cases", [])
        if index >= len(cases):
            state["active"] = False
            self._write_json_path(state_path, state)
            return {}
        case = dict(cases[index])
        state["next_index"] = index + 1
        state["active"] = state["next_index"] < len(cases)
        self._write_json_path(state_path, state)
        return {
            "benchmark_index": index + 1,
            "benchmark_kind": state.get("kind", "benchmark"),
            "benchmark_session_id": state.get("session_id"),
            "ground_truth": case["ground_truth"],
            "condition": case["condition"],
            "repeat": case["repeat"],
            "critical_terms": case["critical_terms"],
        }

    @staticmethod
    def _write_json_path(path: Path, value: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def start_sample(self, raw, metadata: dict | None = None) -> str:
        """Persist raw audio before denoising or ASR can reject the utterance."""
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            slot = self._counter % self.slots
            self._counter += 1
            key = f"sample-{slot:02d}"
            self._write_audio(self.directory / f"{key}-raw.wav", raw)
            filtered_path = self.directory / f"{key}-rnnoise.wav"
            if filtered_path.exists():
                filtered_path.unlink()
            record = {
                "schema_version": 1,
                "sample_id": key,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ground_truth": None,
                "condition": "unlabeled",
                "critical_terms": [],
                "status": "raw_saved",
                "audio": {
                    "raw": f"{key}-raw.wav",
                    "rnnoise": None,
                    "sample_rate": raw.sample_rate,
                    "channels": raw.channels,
                },
                "asr": {"primary_source": None, "raw": None, "rnnoise": None},
            }
            if metadata:
                record.update(metadata)
            record.update(self._claim_benchmark_case())
            self._write_metadata(key, record)
            return key

    def save_filtered(self, key: str, filtered) -> None:
        with self._lock:
            self._write_audio(self.directory / f"{key}-rnnoise.wav", filtered)
            path = self._metadata_path(key)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["audio"]["rnnoise"] = f"{key}-rnnoise.wav"
            record["status"] = "audio_saved"
            self._write_metadata(key, record)

    def update_result(self, key: str, **updates) -> None:
        with self._lock:
            path = self._metadata_path(key)
            record = json.loads(path.read_text(encoding="utf-8"))
            record.update(updates)
            self._write_metadata(key, record)

    def save_pair(self, raw, filtered) -> str:
        key = self.start_sample(raw)
        self.save_filtered(key, filtered)
        return key


def configure_diagnostics_logging(enabled: bool) -> None:
    if not enabled:
        return
    directory = PROJECT_ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("app")
    path = directory / "voice-session.log"
    if any(isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == path for handler in logger.handlers):
        return
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
