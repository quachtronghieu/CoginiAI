# Xiaozhi Backend

Local/offline-first FastAPI backend for Xiaozhi ESP32. No Xiaozhi, Tenclass, OpenAI, EdgeTTS, or other cloud AI service is used.

```text
ESP32 Opus 16 kHz -> PCM -> Silero VAD -> faster-whisper language detection
-> local Ollama/Gemma -> Vietnamese or English Piper voice
-> Opus 24 kHz/60 ms -> ESP32
```

## Windows setup

Python `3.10 - 3.14` is supported on Windows. When moving or copying the project to a new machine, recreate or update `pyvenv.cfg` so the Python executable and paths match the new system.

```powershell
cd D:\4\xiaozhi-team-source\xiaozhi-backend
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Model binaries and `.env` are ignored by Git.

## Configure ESP32 connection

Set the backend LAN IP in `.env`. Never use `localhost` for an ESP32 connection.

```env
PUBLIC_WS_URL=ws://192.168.1.217:8000/xiaozhi/v1/
```

Firmware OTA config must point to:

```text
http://192.168.1.217:8000/xiaozhi/ota/
```

Endpoints:

- OTA: `GET|POST /xiaozhi/ota/`
- WebSocket protocol v1: `/xiaozhi/v1/`
- Server health: `/health`
- Local AI status: `/health/ai`

## ASR: faster-whisper

Backend uses `ASR_LOCAL_FILES_ONLY=true`, so normal startup never downloads a model. Download/cache once with an explicit command while Internet is available:

```powershell
python scripts\setup_asr.py small
```

Then configure:

```env
ASR_MODEL=small
ASR_LANGUAGE=
ASR_COMPUTE_TYPE=
ASR_LOCAL_FILES_ONLY=true
LANGUAGE_HINT_MIN_PROBABILITY=0.80
MODEL_DEVICE=auto
```

Default Hugging Face cache is under the current Windows user cache directory. `MODEL_DEVICE=auto` selects CUDA through CTranslate2 when available, otherwise CPU. CPU defaults to `int8`; CUDA defaults to `float16`.

Blank `ASR_LANGUAGE` uses the selected VI/EN session language when present;
only a fresh session without a language uses automatic detection. Explicit
`ASR_LANGUAGE` remains an operator override. Transcript text is the primary
response-language signal. Forced-language results expose zero detection
confidence; the model's original value is retained separately as
`reported_language_probability` in candidate diagnostics.

If automatic detection is low-confidence, outside Vietnamese/English, or returns
incompatible Cyrillic text, backend retries with `language=en` and `language=vi`.
Candidates are ranked by segment log probability minus no-speech probability;
accents and language identification probability add no selection bonus. A
selected session language avoids cross-language retries. Hotwords are supplied
only for the pending allergy/confirmation question; an empty context does not
reactivate the global list. Empty/noise transcripts are logged and ignored.

## LLM: local Ollama/Gemma

Install Ollama locally, then explicitly download Gemma:

```powershell
ollama pull gemma3:4b
ollama serve
```

If the model is stored in this workspace's offline cache, start Ollama with
that cache visible to the server:

```powershell
$env:OLLAMA_MODELS = "D:\CODE_ESP32S3\.ollama\models"
$env:OLLAMA_HOST = "127.0.0.1:11434"
ollama serve
```

Configure:

```env
LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=gemma3:4b
LLM_NUM_PREDICT=1024
LLM_NUM_CTX=8192
LLM_TIMEOUT_SECONDS=90
AI_PIPELINE_TIMEOUT_SECONDS=120
```

Backend calls Ollama's local `/api/chat` endpoint for short voice responses.

The selected model does not own the deterministic greeting, allergy confirmation,
scope guards, region memory or raw/filtered ASR comparison. These stay in backend
code. Changing `LLM_MODEL` does not rebuild or replace the cuisine index. Still
rerun unit tests, `scripts/check_cuisine_dialogue.py` and the live synthetic voice
smoke test after switching: evidence selection, latency and GPU use can change.
Keep `qwen2.5:3b` installed for rollback by restoring that `LLM_MODEL` value and
restarting the backend. On the 6 GB laptop GPU, test with Whisper resident too;
loading Gemma alone does not establish voice-pipeline memory headroom.

## Cuisine RAG

Build the curated SQLite index from the crawler's `articles_vietnam` directory:

```powershell
cd D:\CODE_ESP32S3\src_new\xiaozhi-team-source
py .\rag-tools\cuisine_rag.py build `
  --data-dir C:\Users\Hieu\Downloads\cogniai-data\tasteofvietnam_data `
  --extra-data-dir .\rag-tools\supplemental `
  --database .\rag-tools\cuisine_v3.sqlite3
```

The supplemental directory is populated only from the explicit, audited URL
manifest. The crawler checks `robots.txt`, expected place names, dish names and
topics before writing an article:

```powershell
py .\rag-tools\targeted_crawler.py `
  --manifest .\rag-tools\supplemental\targets.json `
  --output .\rag-tools\supplemental `
  --existing-database .\rag-tools\cuisine_v3.sqlite3 `
  --report .\rag-tools\supplemental\crawl-report.json
```

Generate the current-province, former-province and tourist-destination coverage
report after every rebuild:

```powershell
cd .\xiaozhi-backend
.\.venv\Scripts\python.exe -m scripts.audit_cuisine_coverage `
  --database ..\rag-tools\cuisine_v3.sqlite3 `
  --json ..\logs\cuisine-coverage-v3.json `
  --markdown ..\logs\cuisine-coverage-v3.md
```

Configure the backend with an absolute database path:

```env
RAG_ENABLED=true
RAG_DB_PATH=D:\CODE_ESP32S3\src_new\xiaozhi-team-source\rag-tools\cuisine_v3.sqlite3
RAG_TOP_K=3
RAG_STRICT=true
```

`RAG_STRICT=true` makes the assistant decline questions that the indexed
articles cannot support. After rebuilding the index, restart the backend and
confirm that `/health/ai` reports `rag: true`.

## Bilingual Vietnamese/English behavior

```text
Vietnamese transcript -> configured LLM answers Vietnamese -> Piper VI
English transcript    -> configured LLM answers English    -> Piper EN
```

For mixed Vietnamese/English input, the configured LLM uses the dominant language unless the user explicitly requests another response language. Explicit requests override every detected hint. Piper routes from actual answer text first, including short English replies such as `Hi` and `OK`. It only falls back to a high-confidence Whisper hint, then `TTS_DEFAULT_LANGUAGE`.

## TTS: local Piper

Download both Piper `.onnx` voices and matching `.onnx.json` configs manually. Keep them outside Git, then set absolute local paths:

```env
TTS_MODEL_VI=D:\CODE_ESP32S3\xiaozhi-backend\models\piper\vi_VN-vais1000-medium.onnx
TTS_MODEL_EN=D:\CODE_ESP32S3\xiaozhi-backend\models\piper\en_US-lessac-medium.onnx
TTS_DEFAULT_LANGUAGE=vi
```

Vietnamese and English voices are cached independently and loaded only on first use. Legacy `TTS_MODEL`/`TTS_VOICE` remains supported when bilingual model variables are unset. Piper output is normalized and encoded as signed 16-bit mono Opus at 24 kHz, 60 ms per packet.

WebSocket audio frames are paced at their configured frame duration. Backend sends `tts/stop` only after the final frame's expected playback interval has drained. An incoming `abort` cancels pacing and sends `tts/stop` immediately.

## RNNoise microphone suppression

Daily use defaults to raw audio and a single ASR service request per utterance:

```env
RNNOISE_ENABLED=false
RNNOISE_ASR_MODE=raw
```

Raw mode bypasses RNNoise processing and comparison, even if an old environment
still sets `RNNOISE_ENABLED=true`. It saves raw WAV/JSON when diagnostics are
enabled; it does not create a misleading duplicate `-rnnoise.wav`. Existing
language/safety retries inside the ASR adapter can still run when needed.

RNNoise has three routing modes. During a separate comparison, use `shadow`: raw audio
drives VAD, Whisper, language selection and the conversation; RNNoise is still
run and transcribed with the same Whisper service only for diagnostics.

```env
RNNOISE_ENABLED=true
RNNOISE_ASR_MODE=shadow
VOICE_DIAGNOSTICS_ENABLED=true
VOICE_DIAGNOSTICS_SLOTS=24
```

The ESP32 sends 16 kHz mono audio. `pyrnnoise` resamples it to RNNoise's native
48 kHz/480-sample frames, suppresses noise, and returns it to 16 kHz. Each
complete utterance uses one RNNoise instance so filter state is retained across
its frames. The backend checks the installed runtime reports 48 kHz and 480
samples per native frame. `/health/ai` reports `rnnoise: true` when the native
library is loaded and active.

`RNNOISE_ASR_MODE=filtered` restores RNNoise before VAD/Whisper.
`RNNOISE_ASR_MODE=raw` uses only raw audio for recognition. In `shadow` mode,
the RNNoise result cannot change the response or customer profile. Shadow runs
the ASR service twice and waits for both results, increasing response time.
See [AUDIO_MODES.md](AUDIO_MODES.md) for daily use, explicit shadow testing,
turbo trials and rollback to small.

### Controlled 24-sample voice comparison

Start the prepared matrix (four exact phrases, quiet/normal/motor, two repeats):

First enable the shadow settings above and restart the backend. The benchmark
commands below use `logs/voice`; an isolated `run_asr_trial` instead writes to
its own directory. Planned sentences must be checked against the actual WAV
before treating them as verified references; extra replies can shift labels.

```powershell
python scripts\voice_benchmark.py begin
```

Speak each printed sentence exactly as shown. Every received utterance gets the
next expected sentence, condition, critical terms, raw WAV, RNNoise WAV and both
Whisper results under the same `sample-NN` ID. Check the next expected sentence
or print the final comparison with:

```powershell
python scripts\voice_benchmark.py status
python scripts\voice_benchmark.py report
```

The report compares character error rate, empty transcripts and exact critical
terms such as `đậu phộng`, `tôm` and `không`. Diagnostic files stay local under
`logs/voice` and rotate within `VOICE_DIAGNOSTICS_SLOTS`; they contain private
audio. Run `python scripts\voice_benchmark.py stop` to stop assigning benchmark
answers to later conversations.

## VAD and limits

Silero loads once; each WebSocket gets isolated iterator and Opus decoder state. Timeout remains a hard safety fallback.

```env
VAD_ENABLED=true
VAD_THRESHOLD=0.6
VAD_STRONG_SPEECH_MS=250
ASR_NO_SPEECH_GATE_PROBABILITY=0.60
VAD_MIN_SILENCE_MS=1000
VAD_SPEECH_PAD_MS=200
MAX_AUDIO_SECONDS=30
MAX_AUDIO_BYTES=2097152
AUTO_STOP_TIMEOUT_SECONDS=6
```

These are the current baseline, shared by code defaults and `.env.example`.
Validate quiet speech for each microphone/distance before changing the VAD
threshold. Explicit `speech_detected=false` rejects the utterance before ASR,
dialogue and TTS. Short speech is not rejected solely for its duration: the
secondary gate combines weak VAD evidence with high ASR no-speech probability.
ASR transcription errors should not be addressed by raising the VAD threshold.

## Run

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/ai
```

`tts`, `tts_vi`, and `tts_en` must all be `true` for bilingual operation. `vad` should also be `true` before ESP32 conversation testing.

## Test AI layers

These commands do not download models:

```powershell
python scripts\test_ai.py check
python scripts\test_ai.py asr .\sample-16k-mono.wav
python scripts\test_ai.py llm "Xin chao" --language vi
python scripts\test_ai.py tts "Xin chao" .\tts-output-vi.wav --language vi
python scripts\test_ai.py tts "Hello" .\tts-output-en.wav --language en
```

Run unit tests:

```powershell
python -m pytest tests -q
```

## ESP32 test

1. Start Ollama and backend.
2. Confirm OTA returns HTTP 200 and the correct LAN WebSocket URL.
3. Confirm `/health/ai` reports ready components.
4. Reboot ESP32 and speak after WebSocket connects.
5. Verify event order: STT, LLM, TTS start, sentence start, binary Opus, TTS stop.
6. Send `abort` or speak again; previous output must stop.

Offline test: download all models first, disconnect Internet but keep Wi-Fi/LAN, then repeat the conversation test.

## Voice safety and source grounding

The backend reads back allergy disclosures and commits them only after a separate
`Đúng` / `Yes`. `Không` during confirmation means the transcription was wrong;
it does not clear previously confirmed allergies. Unrecognised ingredient words
require the complete list to be repeated; there is no silent fuzzy correction.
Additional confirmed allergens are retained conservatively. `Bắt đầu lại` resets
the profile; changing dishes does not. Profiles last for one WebSocket connection,
not across ESP32 disconnects. The original request and location are stored outside
the short chat history, so onboarding does not lose the meal request.

Ambiguous observed locations such as `Cà Mèo` / `Camel` or `Hồ Chị` require
confirmation. Unknown locations are not silently mapped. Region-constrained
retrieval requires the article title to explicitly concern that location.
Breakfast recommendations require explicit breakfast evidence.

List requests such as `những món ăn ở ...` remain recommendations even if ASR
damages the opening verb. An unrecognized place after a location marker asks
for clarification and cannot fall back to unscoped RAG or the previous city.
The RAG entry point also guards unresolved places when no session is supplied.

In RAG mode, the configured LLM selects IDs of complete source passages using a structured JSON
response. Only backend-selected source text reaches TTS; free model claims and
previous assistant turns are not factual evidence. This trades some conversational
fluency for auditable grounding. It validates source membership, not the truth of
the scraped article. Verified passages remain in Vietnamese; an independently
checked English translation is not implemented for RAG yet. Article prices,
hours and availability are historical references, not live restaurant data.

General regional requests and factual food questions are separate from
personalized allergy-suitable recommendations. The former can read RAG even for
customers with declared allergies, but clearly label descriptions as information,
not suitable choices, and require staff confirmation of ingredients/cross-contact.
Explicit allergy-safety requests and personalized suitable choices still require
the staff-verified menu below. Missing allergen words never prove suitability.
The optional conservative whole-article exclusion remains available to callers;
it is not applied to factual descriptions of dishes that contain allergens.

Only meaningful food requests replace the stored request. Location-only followups
retain meal/budget/diet preferences, not the prior location. Unknown locations ask
for clarification; observed Nha Trang misrecognitions require confirmation, never
silent correction. A new factual question exits stale location clarification.
Short place names cannot change an established conversation language. Explicit
language preferences persist until a full session reset. RAG English translation
remains unimplemented; its existing source-language disclosure still applies.

Breakfast evidence must be substantive source text, not a related-article link.
A single-dish article may link descriptive sentences to its explicit meal evidence;
a list cannot use one breakfast dish to qualify every other dish. Availability
and restaurant prices are not inferred. If only one dish is supported, the backend
does not fabricate two more choices. Group-budget planning and full diet matching
still require structured restaurant menu/portion data.

### Staff-verified menu (no real entries provided yet)

`data/verified-menu.json` is intentionally empty. Fill it only from actual recipes,
sauces, garnishes and staff-confirmed cross-contact information. Do not populate
it automatically from crawled articles or model output. Each entry requires:

- `id`, `name`, `served_in` (restaurant location), `ingredients` (complete list).
- `allergens`, `may_contain`: canonical labels from `app/allergy.py`.
- `ingredients_verified`, `allergens_verified`, `cross_contact_assessed`: real
  booleans; all must be `true` only after staff verification.
- `verified_by`, `verified_on` (YYYY-MM-DD); records older than
  `VERIFIED_MENU_MAX_AGE_DAYS` (default 30) or future-dated records are excluded.
- `availability`: `available`, `unavailable` or `unknown`; only available items
  may be suggested. Reconfirm availability with staff before ordering.
- `meal_types`: e.g. `["breakfast"]`; `dietary_tags`: e.g. `["vegetarian"]` only
  if verified; `spicy`: boolean when known; optional `price_vnd`: positive integer.

This is a local trusted-data contract, not an independent certification. Even a
matching menu entry must not be described as guaranteed safe. Complex group-budget
selection is not implemented; missing portion/pricing data is not guessed.

### Bounded private diagnostics and ASR comparison

Greeting/domain/allergy order is deterministic before retrieval/model generation:
`Xin chào` -> greeting and ask what dish is wanted; a recommendation -> ask
allergies if unknown; a complete ingredient list -> read back the WHOLE list;
`chính xác` -> commit it and resume the saved food request. Standalone keywords
(`đậu phộng, sữa`) are accepted while answering the allergy question. Courtesy
and identified non-culinary requests do not change a pending allergy list.
Confirmed allergies last for the connection, not just the short LLM history.

Disclosure prefixes are separated from ingredient names. Known names after an
ASR-damaged first-person introduction can be read back in allergy onboarding;
unparsed ingredient words are named and require a complete-list clarification,
not silently discarded or corrected. Vietnamese tone collisions such as `của`
vs `cua`, `cả` vs `cá` and `đừng` vs `đúng` are not confirmations/ingredient aliases.

In explicit `filtered` mode, RNNoise output is compared with raw speech during allergy onboarding/readback.
Different complete lists or a possible missing ingredient cause clarification.
Disagreeing yes/no recordings cannot commit a list. A clear raw `sữa` can rescue
a damaged filtered word, but the entire list STILL needs human confirmation.
Uncertain short courtesy phrases also have a limited raw-audio fallback. ASR
hotwords depend on dialogue phase; repetitive hotword hallucinations retry once
without hints, then are rejected if still invalid. This is not a guarantee of
accurate recognition: both recordings can miss the same word, so users must
listen to the complete readback and reject it when incomplete. Daily `raw` mode
uses the raw transcript and still requires readback confirmation; `shadow`
records the second transcript without letting it alter the conversation.

Set `VOICE_DIAGNOSTICS_ENABLED=true` to keep rotating `logs/voice-session.log`
and up to `VOICE_DIAGNOSTICS_SLOTS` samples (default 24) under `logs/voice/`.
Raw is saved before filtering or ASR, including VAD-rejected silence. Raw mode
saves raw WAV/JSON only; shadow adds the RNNoise WAV and both ASR results.
ASR metadata includes every candidate, actual hotwords, forced/auto language,
reported versus effective language probability, acoustic metrics and score.
Completed samples include per-stage `timings_ms`, answer, region, route and
sources. Synthesis/encoding time excludes socket pacing and speaker playback.
Samples rotate, so archive speaker-verified recordings before a slot is reused.
Conversation logs include
the original transcript, resolved request, route, language, allergens and selected
source URLs, so physical-device tests can separate ASR from retrieval problems.
These contain private
speech, remain local, are not served over HTTP, and are ignored by Git. Disable
this setting when monitoring is no longer needed. Existing files remain until
you delete them. The ASR no longer overwrites `debug_recorded.wav` on every turn.

```powershell
.venv\Scripts\python.exe scripts\compare_asr.py --model small --audio logs\voice\sample-00-raw.wav logs\voice\sample-00-rnnoise.wav --reference "Words confirmed by the speaker"
.venv\Scripts\python.exe scripts\compare_asr.py --model medium --audio logs\voice\sample-00-raw.wav logs\voice\sample-00-rnnoise.wav --reference "Words confirmed by the speaker"
.venv\Scripts\python.exe scripts\check_cuisine_dialogue.py
.venv\Scripts\python.exe scripts\check_voice_pipeline.py
.venv\Scripts\python.exe scripts\check_recorded_voice.py
.venv\Scripts\python.exe scripts\check_voice_quality.py
```

These commands use cached models only. Without a speaker-checked reference, the
comparison reports transcripts and latency, not accuracy. Language probability
is language identification confidence, not a word-accuracy percentage. ASR hotwords
are hints only. Near-silence and invalid no-speech/log-probability candidates are
rejected before they can update a customer profile. Whisper `small` remains the
configured model pending representative speaker-verified measurements.
Highly repetitive transcripts are rejected even when language confidence is high.
With Silero active, the idle timeout no longer cuts ongoing speech; VAD silence/end
or the hard audio buffer limits finish it. Timeout-only fallback behaviour is
unchanged. The WebSocket smoke test uses synthetic speech, not the ESP microphone.
The clean/noisy quality check reports correct recognition separately from safe
clarification. A passing safety contract does NOT mean every word was recognized.

## Docker

Docker is optional. Local model files and Ollama access must be mounted/configured explicitly.

```powershell
docker compose up --build
```
