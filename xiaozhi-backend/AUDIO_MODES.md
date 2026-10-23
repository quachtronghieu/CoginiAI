# Chế độ nghe của CoginiAI

Các lệnh dưới đây chạy trong thư mục `xiaozhi-backend`, sau khi kích hoạt môi
trường Python đã cài dependencies. Đây là thay đổi backend; không cần nạp lại
firmware ESP32. Mỗi lần chỉ chạy một backend trên cổng 8000.

## Sử dụng bình thường

Cấu hình trong `.env` và `.env.example`:

```env
ASR_MODEL=small
RNNOISE_ENABLED=false
RNNOISE_ASR_MODE=raw
VAD_ENABLED=true
VAD_THRESHOLD=0.6
VAD_STRONG_SPEECH_MS=250
ASR_NO_SPEECH_GATE_PROBABILITY=0.60
VAD_MIN_SILENCE_MS=1000
VAD_SPEECH_PAD_MS=200
AUTO_STOP_TIMEOUT_SECONDS=6
```

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

VAD kiểm tra raw để kết thúc lượt nghe. Nếu VAD xác định không có tiếng nói,
backend lưu chẩn đoán khi được bật, đóng lượt nghe bình thường và bỏ qua
Whisper, hội thoại, TTS. Câu ngắn không bị loại chỉ vì độ dài; bộ chặn thứ hai
kết hợp bằng chứng VAD yếu với xác suất không có tiếng nói cao từ ASR.

Sau khi qua VAD, raw đi vào dịch vụ ASR một lần, bỏ qua RNNoise và lượt ASR
so sánh. ASR vẫn có thể thử lại ngôn ngữ hoặc bỏ hotwords khi kết quả chưa rõ.
Thông tin dị ứng vẫn được đọc lại và chỉ lưu sau xác nhận của người dùng.

`0.6` là cấu hình hiện tại, cần kiểm tra giọng nhỏ theo mic và khoảng cách sử
dụng. Không tăng VAD để sửa lỗi sai chữ. Khi đổi `.env`, khởi động lại backend.
Trong raw mode, `/health/ai` báo `rnnoise: false` là bình thường.

## So sánh raw với RNNoise khi cần

Để so sánh trực tiếp, dừng backend thường bằng Ctrl+C rồi chạy:

```powershell
python -m scripts.run_asr_trial --model small --rnnoise-mode shadow
```

Lệnh tự bật RNNoise và diagnostics cho tiến trình thử, giữ raw làm nguồn
hội thoại, ghi bản raw/RNNoise và hai kết quả ASR. Shadow chờ cả hai lượt
ASR nên có thể phản hồi chậm hơn. `.env` không bị sửa. Nhật ký nằm trong
`logs/asr-trial-<thời gian>/`; thư mục được in khi khởi động.

Để dùng bộ thu 24 mẫu `scripts/voice_benchmark.py` với `logs/voice`, đặt
`RNNOISE_ENABLED=true`, `RNNOISE_ASR_MODE=shadow`,
`VOICE_DIAGNOSTICS_ENABLED=true` trong `.env` rồi chạy backend thường.
Sau đợt đo, trả cấu hình về raw và khởi động lại. Không dùng bộ gán nhãn
`logs/voice` cho thư mục thử cô lập vì chúng là hai nơi lưu khác nhau.

Mỗi mẫu cần ghi nguyên văn câu thực sự đã nói, điều kiện thu và xác nhận
đúng cặp WAV/JSON. Câu theo kịch bản chỉ là nhãn dự kiến cho đến khi nghe
đối chiếu; đừng tính độ chính xác cho mẫu chưa được xác nhận. Kiểm tra riêng
địa danh, tên chất gây dị ứng và chữ "không".

## Thử turbo và quay lại small

Chỉ thử khi đã có model cục bộ `models/faster-whisper-large-v3-turbo`, CUDA
phù hợp và Ollama đang chạy với model trong `.env` (hiện là `gemma3:4b`).
Giữ nguyên Gemma, VAD và các tham số giải mã giữa hai lượt để so model.
Dừng backend hiện tại trước mỗi lần chuyển:

```powershell
python -m scripts.run_asr_trial --model small
```

Sau khi ghi nhận kết quả small, dừng bằng Ctrl+C rồi chạy turbo:

```powershell
python -m scripts.run_asr_trial --model turbo
```

Hai lệnh mặc định dùng raw, tắt RNNoise so sánh. Turbo dùng `int8_float16`,
small dùng `float16`; đây là các cấu hình thử CUDA, không phải so sánh riêng
kiến trúc model với cùng kiểu tính toán. Chỉ định cùng kiểu tính toán khi
chạy `scripts/compare_asr.py` để đo tác động riêng của model trên WAV đã lưu.
Launcher nạp Gemma và yêu cầu `keep_alive=-1`, nạp Whisper, rồi ghi
`config.json`, `session.log`, `gpu.jsonl` và mẫu âm thanh trong thư mục thử.
GPU telemetry là tổng VRAM của thiết bị lấy mẫu mỗi 2 giây, không phải
VRAM riêng của từng model hoặc đỉnh tức thời.

Đánh giá trên cùng WAV đã xác nhận: chào hỏi, địa danh, dị ứng, câu phủ định,
giọng nhỏ, im lặng. Với robot thật, đo cả thời gian từ lúc kết thúc lời nói
đến khi bắt đầu phản hồi; độ trễ replay WAV không đại diện cho toàn bộ robot.
Im lặng phải bị chặn trước ASR/hội thoại/TTS. Chỉ chuyển mặc định sang turbo
khi nhận đúng hơn, giữ được giọng nhỏ và đáp ứng thời gian cùng bộ nhớ GPU.

Để replay các WAV đã lưu với cùng kiểu tính toán `int8_float16` cho cả hai
model, dừng backend, giữ Gemma đã nạp rồi chạy:

```powershell
python -m scripts.replay_voice_models --voice-dir logs/asr-trial-20261023-173627/voice --compute-type int8_float16
```

Thay `--voice-dir` bằng nơi chứa cặp WAV/JSON của lượt cần kiểm tra. Script
chụp bản sao đầu vào và hash, chạy lần lượt hai model trong hai tiến trình,
giữ raw/VAD hiện tại và ghi báo cáo riêng `logs/voice-replay-<thời gian>/`.
Nó chạy qua bộ chặn `_start_inference`, ASR, hội thoại/RAG/Gemma, TTS và mã hóa
thật; không tái tạo thời gian gói Opus, phát loa hay tương tác vật lý.

Mỗi WAV dùng một hồ sơ tiếng Việt mới. Thêm `--allergy-known` để mô phỏng đã
xác nhận không dị ứng và đo đường RAG; đây là hồ sơ kiểm thử, không sửa hồ sơ
khách thật. Các bản thu không có đáp án người nói xác nhận chỉ được so chữ
và thời gian, không tính độ chính xác. Mẫu im lặng và bộ thành phần dị ứng
cũ chỉ được dùng khi có dữ liệu xác nhận tương ứng. Không chuyển mặc định
sang turbo chỉ vì câu nhận dạng nghe có vẻ hợp lý hơn.

Quay lại small: dừng tiến trình thử và chạy backend thường bằng lệnh ở phần
đầu; `.env` vẫn giữ `ASR_MODEL=small`. Khởi động lại ESP chỉ cần nếu thiết bị
chưa tự kết nối lại sau khi backend đổi tiến trình.

## Độ trễ và phát từng câu

`TTS_WARMUP_ENABLED=true` nạp cả hai giọng Piper và tổng hợp thử trước khi
backend nhận kết nối. Đổi lại, khởi động sẽ lâu hơn nhưng lượt hỏi đầu không
phải chờ nạp giọng. Cache âm thanh theo đúng văn bản và giọng, tối đa 32 mục
và 16 MiB; kết quả chọn món/RAG không được cache hay dùng lại giữa khách.

`TTS_SENTENCE_STREAMING=true` phát từng đoạn của câu trả lời đã được kiểm tra
đầy đủ. Backend chuẩn bị trước tối đa một đoạn trong lúc gửi đoạn hiện tại;
một lượt vẫn chỉ có một `tts start/stop`. Ngôn ngữ giọng được chọn một lần
cho toàn câu trả lời. Hủy lượt sẽ hủy đoạn chuẩn bị; lỗi giữa lượt đóng kênh
bình thường để ESP trở về trạng thái chờ. Đặt biến này thành `false` và khởi
động lại backend để quay về tổng hợp toàn bộ câu trả lời trước khi gửi.
Phép tổng hợp native đã bắt đầu có thể hoàn tất trong nền sau khi tác vụ bị
hủy; kết quả đó không được gửi, và truy cập Piper vẫn được tuần tự hóa.

JSON mẫu có `timings_ms.first_audio_ready`, `network_timings_ms` và mã phiên.
`finalize_to_first_audio_sent_ms` đo từ chốt lượt đến gói Opus đầu được gửi.
`estimated_speech_end_to_first_audio_sent_ms` ước tính từ khung VAD có tiếng nói
cuối cùng backend nhận; không phải đo trực tiếp loa robot hoặc thời gian mạng
một chiều. Với streaming, `timings_ms.total` gồm cả chờ gửi âm thanh có nhịp;
thời gian TTS/encode có thể chồng lên thời gian gửi nên không cộng trực tiếp.
`delivery_status` phân biệt đang tổng hợp, đã gửi, sẵn sàng và bị hủy. Âm thanh
đã gửi chưa chứng minh người dùng đã nghe đủ trên loa.

Đo trên cùng nội dung trả lời, với Piper đã làm nóng và cache văn bản trống:

```powershell
python -m scripts.benchmark_response_audio --replay-dir logs/voice-replay-20261023T114337Z
```

Lệnh này loại ASR/Gemma và phát loa khỏi phép đo để kiểm tra riêng lợi ích
tổng hợp từng câu. Replay toàn luồng có thể thêm `--sentence-streaming` để
ghi thời gian đoạn đầu sẵn sàng, vẫn không tái hiện phát loa.

Đối chiếu thời gian chờ kết thúc câu trên xác suất VAD đã ghi:

```powershell
python -m scripts.audit_saved_vad --voice-dir logs/asr-trial-20261023-173627/voice
```

Script so 1000/800/700 ms, giữ ngưỡng nhận giọng 0.6 và không sửa WAV/JSON gốc.
Chưa đổi mặc định 1000 ms vì thiếu mẫu có đáp án về giọng nhỏ và ngập ngừng;
17 vệt xác suất lưu sẵn chưa đủ để khẳng định 700–800 ms giữ được mọi câu.

Thông tin dị ứng nghe chưa rõ nhiều lần được giữ ở trạng thái chưa xác nhận,
không tự lưu danh sách một phần. Bot có thể tiếp tục cung cấp thông tin món
kèm lưu ý chưa xác nhận; không chọn menu theo danh sách chưa đầy đủ. Giá,
thành phần và tình trạng phục vụ xác minh vẫn cần dữ liệu thật của nhà hàng;
không tự điền `data/verified-menu.json` từ suy đoán hoặc bài viết phổ thông.
