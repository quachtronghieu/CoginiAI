"""Isolated replay through actual websocket handler; PCM replaces lost Opus packets.
Real ASR, RNNoise, VAD, dialogue; TTS probe records calls without external synthesis.
"""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

from app.ai import PcmAudio, build_ai_pipeline
from app.config import get_settings
from app.asr import FasterWhisperAsrService
from app.vad import SileroVadFactory
from scripts.diagnose_saved_whisper import read_wav

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/model-comparison-20260929T231340Z/gate-verification-after-fix.json'


class TraceModel:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []
    def transcribe(self, audio, **kwargs):
        segments, info = self.inner.transcribe(audio, **kwargs)
        segments = list(segments)
        self.calls.append({'kwargs': kwargs, 'language': info.language,
                           'language_probability': info.language_probability,
                           'segments_unmodified': [asdict(s) for s in segments]})
        return iter(segments), info


class TraceEngine:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []
    def __getattr__(self, name):
        return getattr(self.inner, name)
    def generate(self, *args, **kwargs):
        results = self.inner.generate(*args, **kwargs)
        self.calls.append({'return_no_speech_prob': kwargs.get('return_no_speech_prob'),
                           'raw_values': [float(r.no_speech_prob) for r in results]})
        return results


class PcmCodec:
    def decode(self, frame):
        return frame
    def close(self):
        pass
    def create_stream_decoder(self):
        return PcmCodec()
    async def decode_opus(self, frames):
        return PcmAudio(b''.join(frames), 16000)
    async def encode_opus(self, audio):
        return []


async def main():
    import app.websocket as ws
    settings = replace(get_settings(), asr_model=str(ROOT / 'models/faster-whisper-large-v3-turbo'),
                       asr_compute_type='int8_float16', diagnostics_enabled=False)
    service = FasterWhisperAsrService(settings)
    model = service._get_model()
    engine = TraceEngine(model.model)
    model.model = engine
    trace = TraceModel(model)
    service._model = trace
    report = {'limitations': ['Original Opus packets unavailable: exact saved PCM injected at decoder boundary.',
                             'Fresh session; both auto and manual end tested.',
                             'TTS boundary is a probe; no external TTS audio synthesized.'], 'runs': []}
    original_settings_hash = hashlib.sha256((ROOT / '.env').read_bytes()).hexdigest()
    for mode in ('auto', 'manual'):
        row = {'mode': mode, 'dialogue_calls': [], 'tts_calls': [], 'events': []}
        pipe = build_ai_pipeline(settings)
        pipe.codec = PcmCodec()
        pipe.asr = service
        base_generate = pipe.llm.generate
        async def generate(*args, **kwargs):
            entry = {'transcript': args[0]}
            row['dialogue_calls'].append(entry)
            answer = await base_generate(*args, **kwargs)
            entry['answer'] = answer
            return answer
        pipe.llm.generate = generate
        class TtsProbe:
            async def synthesize(self, text, *args):
                row['tts_calls'].append(text)
                return PcmAudio(b'\0\0' * 240, 24000)
        pipe.tts = TtsProbe()
        factory = SileroVadFactory(settings)
        factory.codec = PcmCodec()
        base_create = factory.create
        def create():
            vad = base_create()
            row['_vad'] = vad
            return vad
        factory.create = create
        ws.ai_pipeline = pipe
        ws.vad_factory = factory
        done = asyncio.Event()
        base_run = pipe.run
        async def run(*args, **kwargs):
            row['vad_at_pipeline_entry'] = kwargs.get('vad_evidence')
            try:
                result = await base_run(*args, **kwargs)
                row['pipeline_result'] = asdict(result)
                return result
            except Exception as exc:
                row['pipeline_error'] = repr(exc)
                raise
            finally:
                done.set()
        pipe.run = run
        audio, _ = read_wav(OUT.parent / 'sample-15-raw.wav')
        row['pcm_sha256'] = hashlib.sha256(audio.data).hexdigest()
        trace.calls.clear()
        engine.calls.clear()
        messages = [{'text': json.dumps({'type':'hello', 'version':1, 'transport':'websocket',
                        'audio_params':{'format':'opus','sample_rate':16000,'channels':1,'frame_duration':60}})},
                    {'text': json.dumps({'type':'listen','state':'start','mode':mode})}]
        messages += [{'bytes':audio.data[i:i+1920]} for i in range(0,len(audio.data),1920)]
        messages += [{'text': json.dumps({'type':'listen','state':'stop'})}]
        class ReplaySocket:
            headers = {'device-id':'offline-gate-audit', 'authorization': settings.device_token}
            async def accept(self):
                pass
            async def receive(self):
                if messages:
                    return {'type':'websocket.receive', **messages.pop(0)}
                await asyncio.wait_for(done.wait(), 90)
                await asyncio.sleep(.1)
                return {'type':'websocket.disconnect'}
            async def send_json(self, value):
                row['events'].append(value)
            async def close(self, **kwargs):
                row['close'] = kwargs
                done.set()
        await asyncio.wait_for(ws.xiaozhi_websocket(ReplaySocket()), 100)
        row.pop('_vad', None)
        row['whisper_calls'] = list(trace.calls)
        row['ctranslate2_generate_calls'] = list(engine.calls)
        report['runs'].append(row)
        OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'mode':mode,'whisper_calls':len(trace.calls),'dialogue':row['dialogue_calls'],
                          'tts':row['tts_calls'],'error':row.get('pipeline_error')},ensure_ascii=False),flush=True)
    # Confirmed ingredient SET only (not a verified verbatim sentence).
    audio, _ = read_wav(ROOT / 'logs/regression-voice-20260928-0851/sample-02-raw.wav')
    from app.allergy import CustomerProfile
    result = await service.transcribe(audio, customer_profile=CustomerProfile(allergy_question_asked=True))
    report['confirmed_allergen_set_case'] = {'path':'logs/regression-voice-20260928-0851/sample-02-raw.wav',
        'sha256': hashlib.sha256(audio.data).hexdigest(), 'reference_terms':['hải sản','trứng'],
        'reference_source':'scripts/check_recorded_voice.py: user-reported ingredient set; no verbatim reference',
        'result':asdict(result), 'terms_present':{t:t in result.text.lower() for t in ['hải sản','trứng']}}
    report['env_unchanged'] = original_settings_hash == hashlib.sha256((ROOT / '.env').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['confirmed_allergen_set_case'],ensure_ascii=False))
    print(str(OUT))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(main())
