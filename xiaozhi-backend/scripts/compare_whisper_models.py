"""Offline comparison. Separate process per model; never edits live settings."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def gpu():
    value = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    used, free = value.strip().splitlines()[0].split(',')
    return {'used_mib': int(used), 'free_mib': int(free)}


def ollama():
    with urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=10) as response:
        return json.load(response)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    cases = []
    for path in sorted((ROOT / 'logs/voice').glob('sample-??.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        wav = path.parent / meta['audio']['raw']
        if not wav.exists():
            continue
        shutil.copy2(wav, directory / wav.name)
        shutil.copy2(path, directory / path.name)
        truth = meta.get('ground_truth')
        verified = meta.get('label_source') == 'user_confirmed'
        digest = hashlib.sha256(wav.read_bytes()).hexdigest()
        if digest == '63e9029b2745e100251771625d9e8940a6f37925e1ba69db7e88fceda320a682':
            truth, verified = 'xin chào', True
        cases.append({'wav': wav.name, 'sha256': digest, 'created_at': meta.get('created_at'),
                      'ground_truth': truth, 'verified': verified,
                      'condition': meta.get('condition'), 'vad': meta.get('vad'),
                      'historical_asr': meta.get('asr', {}).get('raw')})
    save(directory / 'manifest.json', cases)
    return cases


def worker(directory, model_name, compute):
    from app.config import get_settings
    from app.asr import FasterWhisperAsrService
    from app.ai import LocalAiPipeline
    from scripts.diagnose_saved_whisper import TracingAsrService, read_wav, run_whisper_once
    settings = replace(get_settings(), asr_model=model_name, asr_compute_type=compute)
    key = ('turbo' if 'turbo' in model_name else model_name) + '-' + compute
    report = {'model': model_name, 'compute_type': compute, 'gemma_start': ollama(),
              'gpu_before_load': gpu(), 'rows': [], 'measurements': 'GPU total dedicated memory, sampled every 0.5s; not per-process allocation'}
    assert any(m['name'] == 'gemma3:4b' for m in report['gemma_start']['models'])
    measurements = []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            try:
                measurements.append(gpu())
            except Exception:
                pass
            stop.wait(.5)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        service = FasterWhisperAsrService(settings)
        start = time.perf_counter()
        model = service._get_model()
        report['load_seconds'] = time.perf_counter() - start
        report['actual_compute_type'] = str(model.model.compute_type)
        tracing = TracingAsrService(settings, model)
        cases = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        audio, samples = read_wav(directory / cases[0]['wav'])
        start = time.perf_counter()
        service._transcribe_sync(audio)
        report['warmup_seconds'] = time.perf_counter() - start
        for case in cases:
            audio, samples = read_wav(directory / case['wav'])
            start = time.perf_counter()
            direct = run_whisper_once(model, samples, settings, language=settings.asr_language or None,
                                     use_hotwords=True, vad_filter=True)
            direct_seconds = time.perf_counter() - start
            tracing.trace.clear()
            start = time.perf_counter()
            selected = tracing._transcribe_sync(audio)
            elapsed = time.perf_counter() - start
            evidence = case.get('vad')
            gate = LocalAiPipeline(None, None, None, None,
                vad_strong_speech_ms=settings.vad_strong_speech_ms,
                asr_no_speech_gate_probability=settings.asr_no_speech_gate_probability)
            row = {'wav': case['wav'], 'verified': case['verified'], 'ground_truth': case['ground_truth'],
                   'direct': direct, 'direct_seconds': direct_seconds,
                   'selected': asdict(selected), 'selection_seconds': elapsed,
                   'candidates': list(tracing.trace), 'duration_seconds': len(samples)/16000,
                   'combined_gate_rejects': gate._weak_vad_and_asr(evidence, selected) if evidence else None,
                   'gate_note': 'Historical VAD evidence reused; absent means full live gate not evaluated'}
            report['rows'].append(row)
            save(directory / (key + '.json'), report)
            print(json.dumps({'model': key, 'wav': case['wav'], 'text': selected.text,
                              'seconds': round(elapsed, 3)}, ensure_ascii=False), flush=True)
        report['gemma_end'] = ollama()
        report['gpu_end'] = gpu()
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        stop.set()
        thread.join(3)
        report['gpu_peak_used_mib'] = max((m['used_mib'] for m in measurements), default=None)
        report['gpu_min_free_mib'] = min((m['free_mib'] for m in measurements), default=None)
        save(directory / (key + '.json'), report)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--directory')
    parser.add_argument('--model')
    parser.add_argument('--compute')
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.directory), args.model, args.compute)
        return
    directory = ROOT / 'logs' / ('model-comparison-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    cases = prepare(directory)
    print(f'REPORT_DIR={directory}; cases={len(cases)}', flush=True)
    for model, compute in [('small', 'float16'), ('small', 'int8_float16'),
                           (str(ROOT / 'models/faster-whisper-large-v3-turbo'), 'int8_float16')]:
        subprocess.run([sys.executable, '-m', 'scripts.compare_whisper_models', '--worker',
                        '--directory', str(directory), '--model', model, '--compute', compute],
                       cwd=ROOT, check=True)
    print(f'COMPLETED={directory}', flush=True)


if __name__ == '__main__':
    main()
