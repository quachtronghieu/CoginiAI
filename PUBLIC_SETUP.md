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
cd D:\coniAI\CoginiAI\xiaozhi-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

4. Edit `.env`:

- Set `PUBLIC_WS_URL` to the computer's LAN address
  (this machine: `ws://192.168.1.13:8000/xiaozhi/v1/`).
- Set `RAG_ENABLED=true` and `RAG_DB_PATH` to the checked-in
  `D:\coniAI\CoginiAI\rag-tools\cuisine_v3.sqlite3` if the RAG database is used.
- Set local model paths after downloading Whisper and Piper models, e.g.
  `TTS_MODEL_VI=D:\coniAI\CoginiAI\xiaozhi-backend\models\piper\vi_VN-vais1000-medium.onnx`
  and `TTS_MODEL_EN=D:\coniAI\CoginiAI\xiaozhi-backend\models\piper\en_US-lessac-medium.onnx`.

5. Start Ollama and the backend:

```powershell
ollama serve
cd D:\coniAI\CoginiAI\xiaozhi-backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/ai
```

## ESP32 firmware

Install ESP-IDF for ESP32-S3 (this machine: `C:\esp\v5.5.5\esp-idf`), select the
board in `idf.py menuconfig`, then build and flash:

```powershell
C:\esp\v5.5.5\esp-idf\export.ps1
cd D:\coniAI\CoginiAI\xiaozhi-esp32
idf.py set-target esp32s3
idf.py menuconfig
idf.py -p COMx build flash monitor
```

Replace `COMx` with the actual serial port. Configure the backend OTA/WebSocket
address for the local network before flashing (this machine:
`http://192.168.1.13:8000/xiaozhi/ota/`). The firmware stores some runtime
settings (including speaker volume) in NVS; do not publish personal Wi-Fi data.
