"""Replay saved WAVs through the real VAD gate/ASR/dialogue/TTS, one GPU model at a time.

Run as a module. Original files and .env are never edited. Requires Gemma loaded
in the configured Ollama and the live backend stopped to avoid GPU contention.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def gpu():
    value = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    used, free, utilization = (int(part.strip()) for part in value.split(","))
    return {"used_mib": used, "free_mib": free, "utilization": utilization}


def ollama(settings):
    with urllib.request.urlopen(settings.llm_base_url.rstrip("/") + "/api/ps", timeout=10) as response:
        result = json.load(response)
    if not any(model["name"] == settings.llm_model for model in result.get("models", [])):
        raise RuntimeError("Load the configured Gemma model before comparing VRAM")
    return result


def prepare(source, output, allergy_known=False):
    output.mkdir(parents=True, exist_ok=False)
    cases = []

    def add(wav, meta, kind, name, profile=None, terms=None):
        if not wav.is_file():
            return
        target = output / "inputs" / kind / wav.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, target)
        verified = meta.get("label_source") == "user_confirmed"
        cases.append({
            "id": name, "wav": str(target.relative_to(output)),
            "original_wav": str(wav.resolve()),
            "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "created_at": meta.get("created_at"), "condition": meta.get("condition"),
            "ground_truth": meta.get("ground_truth") if verified else None,
            "reference_verified": verified,
            "critical_terms": terms or [],
            "profile": profile or ({"response_language": "vi", "allergy_status": "none"} if allergy_known else {"response_language": "vi"}),
            "historical_asr": meta.get("asr", {}).get("raw"),
        })
        save(target.with_suffix(".source.json"), meta)

    for path in sorted(source.glob("sample-*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        raw_name = meta.get("audio", {}).get("raw")
        if raw_name:
            add(source / raw_name, meta, "live", "live-" + path.stem)
    silence = ROOT / "logs/model-comparison-20260929T231340Z/sample-15.json"
    if silence.is_file():
        meta = json.loads(silence.read_text(encoding="utf-8"))
        add(silence.parent / meta["audio"]["raw"], meta, "archive", "confirmed-silence")
    allergy = ROOT / "logs/regression-voice-20260928-0851/sample-02-raw.wav"
    if allergy.is_file():
        from scripts.diagnose_saved_whisper import read_wav
        audio, _ = read_wav(allergy)
        # Only this exact archived PCM has a user-reported ingredient set.
        if hashlib.sha256(audio.data).hexdigest() == "626e6944e9b67adfc7aaea65ab66d782ad81cf2e58539be351b7d001cc198c04":
            add(allergy, {"reference_note": "User-reported ingredient set only, not verbatim speech"},
                "allergy", "confirmed-allergen-set",
                {"response_language": "vi", "allergy_question_asked": True}, ["hải sản", "trứng"])
    if not cases:
        raise ValueError("No saved WAV/JSON pairs found")
    save(output / "manifest.json", cases)


async def worker(output, name, compute_type, sentence_streaming=False):
    from app.ai import PcmAudio, build_ai_pipeline
    from app.allergy import CustomerProfile, _normalized
    from app.config import get_settings
    from app.diagnostics import VoiceDiagnostics
    from app.session import AudioSession, FinalizeReason
    from app.vad import SileroVadFactory
    import app.websocket as ws
    from scripts.diagnose_saved_whisper import read_wav

    settings = replace(get_settings(),
        asr_model=str(ROOT / "models/faster-whisper-large-v3-turbo") if name == "turbo" else "small",
        asr_compute_type=compute_type, asr_local_files_only=True,
        rnnoise_enabled=False, rnnoise_asr_mode="raw", diagnostics_enabled=False,
        tts_sentence_streaming=sentence_streaming)
    ws.get_settings = lambda: settings
    folder = output / name
    folder.mkdir(exist_ok=False)
    handler = logging.FileHandler(folder / "session.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("app").addHandler(handler)
    logging.getLogger("app").setLevel(logging.INFO)
    report = {"model": settings.asr_model, "compute_type": compute_type, "sentence_streaming": sentence_streaming,
        "gemma_start": ollama(settings), "gpu_before": gpu(), "rows": [],
        "limitations": [
            "Saved PCM at decoder boundary, manual stop; original Opus timing is unavailable.",
            "Real ASR/dialogue/RAG/Gemma/Piper and encoding; no socket pacing or speaker playback.",
            "Fresh VI profile per WAV with identical overrides; historical session state is not reconstructed.",
            "Unverified transcripts have no accuracy score; ingredient sets are not verbatim references.",
            "GPU figures are total device VRAM sampled every 0.5 seconds, not per-model allocation."]}
    samples = []
    stop = threading.Event()

    def monitor():
        with (folder / "gpu.jsonl").open("w", encoding="utf-8") as stream:
            while not stop.is_set():
                try:
                    sample = {"at": time.time(), **gpu()}
                    samples.append(sample)
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                except Exception as exc:
                    logging.getLogger("app").warning("GPU telemetry: %s", exc)
                stop.wait(.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        pipe = build_ai_pipeline(settings)
        pipe.diagnostics = VoiceDiagnostics(folder / "voice", slots=1000)
        codec = pipe.codec

        class PcmCodec:
            def decode(self, frame):
                return frame
            def close(self):
                pass
            def create_stream_decoder(self):
                return PcmCodec()
            async def decode_opus(self, frames):
                return PcmAudio(b"".join(frames), 16000)
            async def encode_opus(self, audio):
                return await codec.encode_opus(audio)

        pipe.codec = PcmCodec()
        factory = SileroVadFactory(settings)
        factory.codec = PcmCodec()
        started = time.perf_counter()
        model = pipe.asr._get_model()
        report["load_seconds"] = time.perf_counter() - started
        report["actual_compute_type"] = str(model.model.compute_type)
        cases = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        audio, _ = read_wav(output / cases[0]["wav"])
        started = time.perf_counter()
        await pipe.asr.transcribe(audio, customer_profile=CustomerProfile(response_language="vi"))
        report["asr_warmup_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        await pipe.tts.synthesize("Xin chào", "vi", 0.0)
        report["tts_warmup_seconds"] = time.perf_counter() - started
        current = {}
        original_asr = pipe.asr.transcribe
        original_llm = pipe.llm.generate
        original_tts = pipe.tts.synthesize
        original_decode = model.transcribe

        def count_decode(*args, **kwargs):
            current["whisper_calls"] += 1
            return original_decode(*args, **kwargs)

        async def transcribe(audio, **kwargs):
            current["asr_calls"] += 1
            result = await original_asr(audio, **kwargs)
            current["asr"] = asdict(result)
            return result

        async def generate(*args, **kwargs):
            current["dialogue_calls"] += 1
            return await original_llm(*args, **kwargs)

        async def synthesize(*args, **kwargs):
            current["tts_calls"] += 1
            return await original_tts(*args, **kwargs)

        async def capture_response(socket, session_id, result, *args, **kwargs):
            current["transcript"] = result.transcript
            current["answer"] = (current.get("answer", "") + " " + result.answer).strip()
            current["output_frames"] = current.get("output_frames", 0) + len(result.audio_frames)
            current.setdefault("first_audio_seconds", time.perf_counter() - started)
            if kwargs.get("on_first_frame"):
                kwargs["on_first_frame"](time.monotonic())

        class Socket:
            async def close(self, **kwargs):
                current["close"] = kwargs
            async def send_json(self, value):
                current.setdefault("events", []).append(value)

        model.transcribe = count_decode
        pipe.asr.transcribe = transcribe
        pipe.llm.generate = generate
        pipe.tts.synthesize = synthesize
        ws.ai_pipeline = pipe
        ws._send_ai_result = capture_response
        for index, case in enumerate(cases):
            current = {"id": case["id"], "whisper_calls": 0,
                "asr_calls": 0, "dialogue_calls": 0, "tts_calls": 0}
            path = output / case["wav"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
            audio, _ = read_wav(path)
            session = AudioSession(case["id"], "offline", "offline", 60, 30, 2 * 1024 * 1024, factory.create())
            session.customer_profile = CustomerProfile(**case["profile"])
            session.start("manual")
            for offset in range(0, len(audio.data), 1920):
                session.append_frame(audio.data[offset:offset + 1920])
            started = time.perf_counter()
            task = await ws._start_inference(Socket(), session, FinalizeReason.MANUAL_STOP, None)
            if task:
                await asyncio.wait_for(task, settings.ai_pipeline_timeout_seconds + 10)
            current["response_ready_seconds"] = time.perf_counter() - started
            current["vad"] = session.last_vad_evidence
            current["region"] = session.customer_profile.region
            current["route"] = session.customer_profile.last_route
            current["action"] = session.customer_profile.dialogue_action
            current["sources"] = session.customer_profile.evidence_sources
            meta_path = folder / "voice" / f"sample-{index:02d}.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                current["status"] = meta["status"]
                current["timings_ms"] = meta.get("timings_ms", {})
            if case["critical_terms"]:
                current["critical_terms_present"] = {term: _normalized(term) in _normalized(current.get("transcript", "")) for term in case["critical_terms"]}
            if case["reference_verified"] and case["ground_truth"] == "":
                current["silence_gate_pass"] = all(current[key] == 0 for key in ("whisper_calls", "dialogue_calls", "tts_calls"))
            session.close()
            report["rows"].append(current)
            save(folder / "results.json", report)
            print(json.dumps({"model": name, "id": case["id"], "text": current.get("transcript"),
                "status": current.get("status"), "seconds": round(current["response_ready_seconds"], 3)}, ensure_ascii=False), flush=True)
        report["gemma_end"] = ollama(settings)
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        stop.set()
        thread.join(3)
        report["gpu_peak_used_mib"] = max((row["used_mib"] for row in samples), default=None)
        report["gpu_min_free_mib"] = min((row["free_mib"] for row in samples), default=None)
        save(folder / "results.json", report)


def summarize(output):
    reports = {name: json.loads((output / name / "results.json").read_text(encoding="utf-8")) for name in ("small", "turbo")}
    lines = ["# Saved WAV replay", "", "All new live references are unverified; no general accuracy score is claimed.",
        "Real synthesis/encoding are included; socket pacing, playback and original packet timing are excluded.", "",
        "| Model | Median ASR s | Median response-ready s | Peak total GPU MiB | Min free MiB |",
        "|---|---:|---:|---:|---:|"]
    for name, report in reports.items():
        asr_times = [row["timings_ms"]["asr"] / 1000 for row in report["rows"] if "asr" in row.get("timings_ms", {})]
        response_times = [row["response_ready_seconds"] for row in report["rows"] if row.get("transcript")]
        lines.append(f"| {name} | {statistics.median(asr_times):.3f} | {statistics.median(response_times):.3f} | {report['gpu_peak_used_mib']} | {report['gpu_min_free_mib']} |")
    lines.extend(["", "| WAV | small | turbo |", "|---|---|---|"])
    for left, right in zip(reports["small"]["rows"], reports["turbo"]["rows"]):
        def label(row):
            return (row.get("transcript") or row.get("status", "no response")).replace("|", "/")
        lines.append(f"| {left['id']} | {label(left)} | {label(right)} |")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    save(output / "summary.json", reports)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("small", "turbo"))
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--allergy-known", action="store_true",
                        help="Simulate already-confirmed no allergies to exercise regional RAG/Gemma")
    parser.add_argument("--sentence-streaming", action="store_true",
                        help="Measure sentence synthesis and first audio readiness; socket playback still excluded")
    args = parser.parse_args()
    if args.worker:
        asyncio.run(worker(args.output, args.worker, args.compute_type, args.sentence_streaming))
        return
    if args.voice_dir is None:
        parser.error("--voice-dir is required")
    output = args.output or ROOT / "logs" / ("voice-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    prepare(args.voice_dir, output, args.allergy_known)
    print("REPORT_DIR=" + str(output.resolve()), flush=True)
    for name in ("small", "turbo"):
        command = [sys.executable, "-m", "scripts.replay_voice_models", "--worker", name,
            "--output", str(output.resolve()), "--compute-type", args.compute_type]
        if args.sentence_streaming:
            command.append("--sentence-streaming")
        subprocess.run(command, cwd=ROOT, check=True)
    summarize(output)
    print("COMPLETED=" + str(output.resolve()), flush=True)


if __name__ == "__main__":
    main()
