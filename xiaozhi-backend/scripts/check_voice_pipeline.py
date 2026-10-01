"""Synthetic speech smoke test via a dedicated WebSocket, not the physical ESP microphone."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

from app.codec import PyAvOpusCodec
from app.config import get_settings
from app.tts import create_tts_service


async def main():
    settings = get_settings()
    tts = create_tts_service(settings)
    codec = PyAvOpusCodec(replace(settings, server_sample_rate=16000))
    turns = ["Xin chào", "Thời tiết hôm nay thế nào", "Gợi ý món ăn sáng ở Nha Trang", "Tôi bị dị ứng với hải sản và trứng", "Chính xác", "Gợi ý món ăn sáng ở Hải Phòng", "Nha Trang", "Gợi ý món ăn an toàn phù hợp dị ứng ở Hải Phòng"]
    packets = []
    for text in turns:
        audio = await tts.synthesize(text, "vi", 1.0)
        packets.append(await codec.encode_opus(audio))
    headers = {"Device-Id": "local-synthetic-smoke-test", "Client-Id": "voice-roadmap"}
    if settings.device_token:
        headers["Authorization"] = settings.device_token if " " in settings.device_token else "Bearer " + settings.device_token
    async with websockets.connect(f"ws://127.0.0.1:{settings.port}/xiaozhi/v1/", additional_headers=headers) as socket:
        await socket.send(json.dumps({"type": "hello", "transport": "websocket", "version": 1, "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60}}))
        hello = json.loads(await socket.recv())
        assert hello["type"] == "hello"
        for index, (reference, frames) in enumerate(zip(turns, packets)):
            await socket.send(json.dumps({"type": "listen", "state": "start", "mode": "manual"}))
            for packet in frames:
                await socket.send(packet)
            await socket.send(json.dumps({"type": "listen", "state": "stop"}))
            transcript, answer, got_audio = "", "", False
            async with asyncio.timeout(120):
                while True:
                    message = await socket.recv()
                    if isinstance(message, bytes):
                        if not got_audio:
                            got_audio = True
                            await socket.send(json.dumps({"type": "abort"}))
                        continue
                    event = json.loads(message)
                    if event["type"] == "alert":
                        raise RuntimeError(event)
                    if event["type"] == "stt":
                        transcript = event["text"]
                    if event["type"] == "tts" and event.get("state") == "sentence_start":
                        answer = event["text"]
                    if event["type"] == "tts" and event.get("state") == "stop":
                        break
            assert transcript and answer and got_audio
            if index == 0:
                assert "tư vấn món" in answer and "dị ứng" not in answer, answer
            elif index == 1:
                assert "chỉ trả lời" in answer, answer
            elif index == 2:
                assert "bạn có dị ứng" in answer, answer
            elif index == 3:
                assert "hải sản" in answer and "trứng" in answer and "đúng không" in answer, answer
            elif 4 <= index <= 6:
                city, dish = ("Hải Phòng", "bánh đa cua") if index == 5 else ("Nha Trang", "bún lá cá dầm")
                assert city in answer and dish in answer.casefold(), (transcript, answer)
                assert "không phải gợi ý món phù hợp" in answer and "nhiễm chéo" in answer, answer
                assert "bạn có dị ứng" not in answer, answer
            else:
                assert "Chưa có bảng thành phần" in answer and "nhân viên" in answer, answer
            print(json.dumps({"reference": reference, "transcript": transcript, "answer": answer, "received_opus": got_audio, "session_id": hello["session_id"]}, ensure_ascii=False), flush=True)
        print(json.dumps({"result": "PASS", "synthetic_voice_turns": len(turns), "physical_microphone_tested": False}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
