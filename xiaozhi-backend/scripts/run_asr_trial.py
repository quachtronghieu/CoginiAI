"""Run isolated ASR settings: python -m scripts.run_asr_trial --model turbo|small.
Add --rnnoise-mode shadow only for a separate raw/RNNoise comparison.
Stop the old backend before switching. .env remains unchanged.
"""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['turbo', 'small'], default='turbo')
    parser.add_argument('--rnnoise-mode', choices=['raw', 'shadow'], default='raw',
                        help='raw: one ASR input; shadow: transcribe raw and RNNoise for comparison')
    return parser.parse_args(argv)


def configure_trial_environment(args, root, folder):
    """Apply process-local overrides before app.config reads .env."""
    os.environ['ASR_MODEL'] = str(root / 'models/faster-whisper-large-v3-turbo') if args.model == 'turbo' else 'small'
    os.environ['ASR_COMPUTE_TYPE'] = 'int8_float16' if args.model == 'turbo' else 'float16'
    os.environ['VOICE_DIAGNOSTICS_ENABLED'] = 'true'
    os.environ['VOICE_DIAGNOSTICS_DIR'] = str(folder / 'voice')
    os.environ['VOICE_DIAGNOSTICS_SLOTS'] = '1000'
    os.environ['RNNOISE_ASR_MODE'] = args.rnnoise_mode
    os.environ['RNNOISE_ENABLED'] = 'true' if args.rnnoise_mode == 'shadow' else 'false'


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / 'logs' / ('asr-trial-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True)
    configure_trial_environment(args, root, folder)
    from app.main import app
    from app.websocket import ai_pipeline
    from app.config import get_settings
    settings = get_settings()
    handler = logging.FileHandler(folder / 'session.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.getLogger('app').addHandler(handler)
    logging.getLogger('app').setLevel(logging.INFO)
    config = {'model':settings.asr_model,'compute_type':settings.asr_compute_type,
              'vad_threshold':settings.vad_threshold, 'rnnoise_mode':settings.rnnoise_asr_mode,
              'rnnoise_enabled':settings.rnnoise_enabled, 'model_device':settings.model_device,
              'llm_model':settings.llm_model, 'llm_num_ctx':settings.llm_num_ctx,
              'vad_min_silence_ms':settings.vad_min_silence_ms,
              'tts_warmup_enabled':settings.tts_warmup_enabled,
              'tts_sentence_streaming':settings.tts_sentence_streaming}
    (folder / 'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('TRIAL_LOG_DIR=' + str(folder), flush=True)
    payload = json.dumps({'model':settings.llm_model,'prompt':'','stream':False,
                          'keep_alive':-1,'options':{'num_ctx':settings.llm_num_ctx}}).encode()
    with urllib.request.urlopen(urllib.request.Request(settings.llm_base_url + '/api/generate',
        data=payload,headers={'Content-Type':'application/json'}),timeout=300) as response:
        json.load(response)
    ai_pipeline.asr._get_model()
    stop = threading.Event()
    def monitor():
        with (folder / 'gpu.jsonl').open('a',encoding='utf-8') as output:
            while not stop.is_set():
                try:
                    value = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                                                     '--format=csv,noheader,nounits'],text=True).strip()
                    output.write(json.dumps({'time':datetime.now().isoformat(),'used_mib,free_mib,gpu_percent':value})+'\n')
                    output.flush()
                except Exception as exc:
                    logging.getLogger('app').warning('GPU telemetry: %s',exc)
                stop.wait(2)
    thread = threading.Thread(target=monitor,daemon=True)
    thread.start()
    import uvicorn
    try:
        uvicorn.run(app,host='0.0.0.0',port=8000)
    finally:
        stop.set()
        thread.join(3)


if __name__ == '__main__':
    main()
