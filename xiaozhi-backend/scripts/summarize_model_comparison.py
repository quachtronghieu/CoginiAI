"""Summarize saved model results and replay VAD over decoded PCM snapshots."""
import json
from pathlib import Path
import statistics
import sys
from app.ai import AsrResult, LocalAiPipeline
from app.config import get_settings
from app.vad import SileroVadFactory
from scripts.diagnose_saved_whisper import read_wav


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    folder = Path(sys.argv[1])
    cases = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    settings = get_settings()
    factory = SileroVadFactory(settings)
    gate = LocalAiPipeline(None, None, None, None,
        vad_strong_speech_ms=settings.vad_strong_speech_ms,
        asr_no_speech_gate_probability=settings.asr_no_speech_gate_probability)
    class PcmDecoder:
        def decode(self, frame):
            return frame
        def close(self):
            pass
    vad = {}
    for case in cases:
        session = factory.create()
        session.decoder.close()
        session.decoder = PcmDecoder()
        audio, _ = read_wav(folder / case['wav'])
        for offset in range(0, len(audio.data), 1920):
            session.should_finalize(audio.data[offset:offset+1920], offset / 32000)
        vad[case['wav']] = session.evidence
        session.close()
    results = {}
    for name in ('small-float16', 'small-int8_float16', 'turbo-int8_float16'):
        path = folder / (name + '.json')
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for row in data['rows']:
            result = AsrResult(**row['selected'])
            rejected = gate._weak_vad_and_asr(vad[row['wav']], result)
            row['replayed_combined_gate_rejects'] = rejected
            row['would_reach_dialogue'] = bool(result.text) and not rejected
        results[name] = data
    report = {'note': 'Offline PCM VAD replay, no Opus packet timing recreation; model comparison uses default profile/hotwords, not recovered historic session state.',
              'vad_replay': vad, 'models': results}
    (folder / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Saved WAV model comparison', '',
             'Only sample-00 greeting and sample-15 silence are user-confirmed. Other labels are provisional.', '',
             'Gemma loaded with context 8192 throughout. GPU numbers are total dedicated memory, not per-model allocations.', '',
             '| Model | Load s | Warmup s | Median ASR s | Max ASR s | Peak GPU MiB | Min free MiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, data in results.items():
        times = [r['selection_seconds'] for r in data['rows']]
        lines.append(f"| {name} | {data['load_seconds']:.2f} | {data['warmup_seconds']:.2f} | {statistics.median(times):.3f} | {max(times):.3f} | {data.get('gpu_peak_used_mib')} | {data.get('gpu_min_free_mib')} |")
    lines.extend(['', '| WAV | small FP16 | small INT8/FP16 | turbo INT8/FP16 |', '|---|---|---|---|'])
    for case in cases:
        cells = [case['wav']]
        for name in ('small-float16', 'small-int8_float16', 'turbo-int8_float16'):
            row = next((r for r in results.get(name, {}).get('rows', []) if r['wav'] == case['wav']), None)
            cells.append(row['selected']['text'].replace('|', '/') or '(empty)' if row else '(missing)')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', '## Confirmed silence (sample-15)', ''])
    for name, data in results.items():
        row = next(r for r in data['rows'] if r['wav'] == 'sample-15-raw.wav')
        lines.append(f"- {name}: text={row['selected']['text']!r}; reaches dialogue after combined gate={row['would_reach_dialogue']}; no_speech={row['selected']['no_speech_probability']:.4f}")
    (folder / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
