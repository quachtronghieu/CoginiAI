# Public clone setup

This repository contains two parts:

- `xiaozhi-backend`: FastAPI voice backend, Whisper, RAG, Ollama and TTS.
- `xiaozhi-esp32`: ESP-IDF firmware for the ESP32-S3 audio device.

Local `.env`, downloaded models, build output and diagnostic recordings are
intentionally excluded from Git. Never commit Wi-Fi passwords, device tokens,
or API keys.

## Backend

1. Install Python 3.11 or newer and Ollama.
2. Pull the configured model:

```powershell
ollama pull gemma3:4b
```

3. Create the environment and install dependencies:

```powershell
cd xiaozhi-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

4. Edit `.env`:

- Set `PUBLIC_WS_URL` to the computer's LAN address.
- Set `RAG_ENABLED=true` and `RAG_DB_PATH` to the checked-in
  `../rag-tools/cuisine_v3.sqlite3` if the RAG database is used.
- Set local model paths after downloading Whisper and Piper models.
- Daily audio defaults are `ASR_MODEL=small`, `RNNOISE_ENABLED=false`,
  `RNNOISE_ASR_MODE=raw` and `VAD_THRESHOLD=0.6`. Validate quiet speech with
  the new microphone before tuning the threshold. See
  [audio modes and reversible turbo trials](xiaozhi-backend/AUDIO_MODES.md).

5. Start Ollama and the backend:

```powershell
ollama serve
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/ai
```

## ESP32 firmware

Install ESP-IDF for ESP32-S3, select the board in `idf.py menuconfig`, then
build and flash:

```powershell
cd xiaozhi-esp32
idf.py set-target esp32s3
idf.py menuconfig
idf.py -p COM7 build flash monitor
```

Replace `COM7` with the actual serial port. Configure the backend OTA/WebSocket
address for the local network before flashing. The firmware stores some runtime
settings (including speaker volume) in NVS; do not publish personal Wi-Fi data.
