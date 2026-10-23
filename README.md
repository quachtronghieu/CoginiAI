# Xiaozhi AI Culinary Assistant - Hướng Dẫn Phát Triển & Vận Hành

Dự án trợ lý giọng nói thông minh chuyên biệt **100% về Ẩm thực & Món ăn** (AI Culinary Assistant), chạy kết hợp giữa vi điều khiển **ESP32** (phần cứng Audio/LCD) và **FastAPI Backend** (trên máy tính GPU).

---

## 1. Kiến Trúc Hệ Thống (Pipeline)

```text
[ESP32 Mic] (Opus 16kHz)
      │
      ▼ (WebSocket ws://<IP>:8000/xiaozhi/v1/)
[FastAPI Backend]
      │
      ├── 1. PyAV Decoder (Opus -> PCM 16kHz)
      ├── 2. Silero VAD (Nhận diện khoảng dừng, không cắt câu)
      ├── 3. Faster-Whisper ASR (small, âm thanh raw; CUDA hoặc CPU theo cấu hình)
      ├── 4. Ollama LLM (Gemma 3 4B - Hệ thống Prompt chuyên gia ẩm thực)
      ├── 5. Microsoft Edge-TTS (Giọng tiếng Việt Hoài My / Nam Minh)
      └── 6. PyAV Encoder (PCM 24kHz -> Opus 60ms)
      │
      ▼ (WebSocket)
[ESP32 Speaker / LCD ST7789]
```

---

Chế độ sử dụng bình thường đặt `RNNOISE_ENABLED=false`,
`RNNOISE_ASR_MODE=raw`, `VAD_THRESHOLD=0.6`. RNNoise shadow chỉ bật cho kiểm thử
so sánh vì nó chạy thêm một lượt nhận dạng. Hướng dẫn dùng raw, thử turbo và
quay lại small nằm tại [AUDIO_MODES.md](xiaozhi-backend/AUDIO_MODES.md).

## 2. Hướng Dẫn Đổi IP Mạng Mới (Wi-Fi Mới / Hotspot)

Khi máy tính chuyển sang mạng Wi-Fi mới hoặc phát Hotspot điện thoại (ví dụ: iPhone Hotspot), địa chỉ IP LAN của máy tính sẽ thay đổi.

### Bước 1: Lấy địa chỉ IP mới của máy tính
Mở PowerShell trên máy tính:
```powershell
ipconfig
```
Tìm adapter Wi-Fi đang kết nối, ghi lại dòng **IPv4 Address** (Ví dụ: `192.168.1.50` hoặc `172.20.10.2`).

### Bước 2: Cập nhật địa chỉ IP trong các file

Bạn cần cập nhật IP mới vào **2 file chính sau**:

1. **`xiaozhi-backend\.env`** (Bắt buộc cho Backend):
   Mở file `xiaozhi-backend\.env`, sửa dòng `PUBLIC_WS_URL`:
   ```env
   PUBLIC_WS_URL=ws://<IP_MOI>:8000/xiaozhi/v1/
   ```
   *(Ví dụ: `PUBLIC_WS_URL=ws://192.168.1.50:8000/xiaozhi/v1/`)*

2. **`xiaozhi-esp32\sdkconfig`** (Bắt buộc cho ESP32):
   Mở file `xiaozhi-esp32\sdkconfig`, tìm dòng `CONFIG_OTA_URL` và cập nhật:
   ```properties
   CONFIG_OTA_URL="http://<IP_MOI>:8000/xiaozhi/ota/"
   ```
   *(Ví dụ: `CONFIG_OTA_URL="http://192.168.1.50:8000/xiaozhi/ota/"`)*

3. *(Tùy chọn khi chạy automated test)* **`xiaozhi-backend\tests\conftest.py`**:
   Sửa dòng:
   ```python
   public_ws_url="ws://<IP_MOI>:8000/xiaozhi/v1/",
   ```

### Bước 3: Nạp lại firmware cho ESP32 (Nếu ESP32 chưa nhận IP mới)
Nếu ESP32 chưa lưu cấu hình hoặc cần nạp lại:
```powershell
cd xiaozhi-esp32
idf.py build flash monitor
```

---

## 3. Cài Đặt Trên Máy Tính Có GPU NVIDIA Mới

Dự án sử dụng GPU NVIDIA (RTX 3050, 4060, 3060, 4070...) để chạy **Faster-Whisper ASR** tốc độ cao (< 0.3s) và **Ollama LLM**.

### Bước 1: Cài đặt Driver NVIDIA & CUDA
- Cài đặt bản [NVIDIA Driver](https://www.nvidia.com/download/index.aspx) mới nhất.
- Kiểm tra card đồ họa bằng PowerShell:
  ```powershell
  nvidia-smi
  ```

### Bước 2: Cài đặt Python & Cấu hình Môi trường ảo (.venv)
- Khuyên dùng **Python 3.10, 3.11 hoặc 3.12** (hoặc 3.14 trên Windows).
- Tạo môi trường ảo mới tại thư mục gốc dự án:
  ```powershell
  cd D:\4\xiaozhi-team-source
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

> [!WARNING]
> **LƯU Ý QUAN TRỌNG KHI COPY SANG MÁY KHÁC (Sửa `pyvenv.cfg`):**
> Nếu bạn copy toàn bộ thư mục dự án chứa sẵn `.venv` sang một máy tính khác (hoặc đổi tên thư mục/ổ đĩa), Python sẽ bị lỗi do đường dẫn tuyệt đối trong file `pyvenv.cfg` không khớp với máy mới:
> 
> **Cách xử lý 1 (Sửa trực tiếp `pyvenv.cfg`):**
> Mở file `.venv\pyvenv.cfg` (và `xiaozhi-backend\.venv\pyvenv.cfg` nếu có), dùng lệnh `where python` trong PowerShell để tìm đường dẫn Python máy bạn, sau đó cập nhật lại các dòng:
> ```ini
> home = C:\Users\<Username_Của_Bạn>\AppData\Local\Programs\Python\Python314
> include-system-site-packages = false
> version = 3.14.5
> executable = C:\Users\<Username_Của_Bạn>\AppData\Local\Programs\Python\Python314\python.exe
> command = C:\Users\<Username_Của_Bạn>\AppData\Local\Programs\Python\Python314\python.exe -m venv <Đường_Dẫn_Dự_Án_Hiện_Tại>\.venv
> ```
> 
> **Cách xử lý 2 (Khuyên dùng - Tạo lại venv sạch):**
> ```powershell
> Remove-Item -Recurse -Force .venv
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1
> ```

### Bước 3: Cài đặt PyTorch với hỗ trợ GPU CUDA
Chạy lệnh sau để cài đặt PyTorch CUDA tương thích với CTranslate2:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Kiểm tra PyTorch nhận diện GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```
*(Kết quả in ra `CUDA Available: True` là thành công)*.

### Bước 4: Cài đặt Dependencies Backend
```powershell
cd xiaozhi-backend
pip install -r requirements.txt
```

### Bước 5: Tải Model Faster-Whisper ASR (Offline Cache)
Tải trước model `small` về máy (chỉ cần chạy 1 lần khi có mạng):
```powershell
python scripts\setup_asr.py small
```

---

## 4. Cài Đặt & Cấu Hình Ollama với Model Gemma 3 (4B)

Dự án hiện cấu hình mô hình **`gemma3:4b`**. Đo tốc độ phản hồi và bộ nhớ GPU
khi chạy cùng Whisper trên máy thực tế; không suy ra độ trễ robot từ tên model.

### Bước 1: Cài đặt Ollama
Tải và cài đặt Ollama từ trang chủ: [https://ollama.com](https://ollama.com)

### Bước 2: Kéo model Gemma 3 (4B)
Mở PowerShell chạy lệnh:
```powershell
ollama pull gemma3:4b
```

Kiểm tra danh sách model đã tải:
```powershell
ollama list
```

### Bước 3: Chạy Ollama Service
Đảm bảo service Ollama đang chạy tại cổng mặc định `11434`:
```powershell
ollama serve
```

### Bước 4: Kiểm tra cấu hình trong `xiaozhi-backend\.env`
File `.env` phải có các thông số:
```env
LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=gemma3:4b
LLM_NUM_PREDICT=1024
LLM_NUM_CTX=8192
LLM_TIMEOUT_SECONDS=90
```

---

## 5. Cấu Hình Giọng Đọc (Microsoft Edge-TTS)

Backend sử dụng Microsoft Edge-TTS chất lượng cao, hoàn toàn miễn phí và tự nhiên:
Trong `xiaozhi-backend\.env`:
```env
TTS_ENGINE=edge
TTS_VOICE_VI=vi-VN-HoaiMyNeural
TTS_VOICE_EN=en-US-JennyNeural
TTS_DEFAULT_LANGUAGE=vi
```
- Giọng nữ tiếng Việt: `vi-VN-HoaiMyNeural` (Hoài My - mặc định).
- Nếu muốn đổi sang giọng nam tiếng Việt: đổi thành `vi-VN-NamMinhNeural`.

---

## 6. Hướng Dẫn Khởi Chạy Toàn Bộ Hệ Thống

### Bước 1: Bật Ollama
Đảm bảo biểu tượng Ollama đang chạy ở thanh Taskbar (hoặc chạy lệnh `ollama serve`).

### Bước 2: Khởi động Backend
Mở PowerShell tại thư mục `xiaozhi-backend`:
```powershell
cd D:\4\xiaozhi-team-source\xiaozhi-backend
..\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Khi khởi động thành công, màn hình sẽ hiển thị:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Bước 3: Bật ESP32
- Cắm nguồn cho bo mạch ESP32.
- ESP32 sẽ tự động kết nối Wi-Fi và kết nối WebSocket đến `ws://<IP>:8000/xiaozhi/v1/`.
- Màn hình LCD hiển thị trạng thái sẵn sàng lắng nghe.

---

## 7. Các Vai Trò Ẩm Thực Hỗ Trợ (Voice Commands)

Trợ lý được thiết kế chuyên biệt **100% về lĩnh vực ẩm thực**. Người dùng có thể đổi phong cách nói chuyện bằng giọng nói:

| Lệnh giọng nói | Vai trò chuyển đổi | Đặc điểm phản hồi |
|---|---|---|
| *"Bạn bây giờ là bạn thân nhé"* | **Foodie Friend** (Bạn thân sành ăn) | Xưng hô *"cậu - tớ"*, *"mày - tao"*, vui vẻ, cùng rủ đi ăn quán ngon, ăn vặt. |
| *"Hãy đóng vai đầu bếp"* | **Master Chef** (Đầu bếp chuyên nghiệp) | Xưng hô *"bếp trưởng"*, hướng dẫn chi tiết công thức nấu, nêm nếm gia vị. |
| *"Hãy làm chuyên gia dinh dưỡng"* | **Culinary Nutritionist** | Tư vấn lượng calo, thực đơn lành mạnh, ăn kiêng, tốt cho sức khỏe. |
| *"Quay lại bình thường"* | **Default Culinary AI** | Trợ lý tư vấn ẩm thực chuẩn mực, lịch sự. |
| *"Đổi chủ đề món khác"* | **Xóa lịch sử trò chuyện** | Làm mới ngữ cảnh để bắt đầu gợi ý món ăn mới. |

---

## 8. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### 1. ESP32 không kết nối được tới Backend
- **Kiểm tra Windows Firewall**: Cho phép cổng `8000` qua tường lửa Windows:
  ```powershell
  New-NetFirewallRule -DisplayName "Xiaozhi Backend Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
  ```
- **Kiểm tra cùng mạng**: ESP32 và máy tính phải bắt chung 1 sóng Wi-Fi hoặc chung Hotspot.

### 2. Bị cắt câu khi nói câu dài
Trong `xiaozhi-backend\.env`, cấu hình VAD đã được tối ưu cho câu hỏi dài:
```env
VAD_MIN_SILENCE_MS=800
AUTO_STOP_TIMEOUT_SECONDS=6
```
*(Nếu cần ngắt nghỉ hơi lâu hơn nữa giữa câu, có thể nâng `VAD_MIN_SILENCE_MS=1000`).*

### 3. Lỗi thiếu file DLL CUDA trên Windows khi chạy Faster-Whisper
File `app/asr.py` đã tích hợp cơ chế tự động tìm và nạp DLL từ thư mục `torch/lib` và `nvidia/cublas/bin`. Nếu gặp lỗi `cublas64_12.dll not found`, hãy đảm bảo đã cài đúng bản `torch` CUDA theo Bước 3 ở Mục 3.

### 4. Kiểm tra sức khỏe toàn bộ hệ thống
Mở trình duyệt truy cập:
`http://localhost:8000/health/ai`
Kết quả trả về:
```json
{
  "asr": true,
  "tts": true,
  "tts_vi": true,
  "tts_en": true,
  "llm": true,
  "vad": true
}
```
Nếu tất cả đều `true`, hệ thống đã sẵn sàng 100%!
