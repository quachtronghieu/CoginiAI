import asyncio
import logging
from .diagnostics import supported_kwargs
import secrets
import time
from collections.abc import Callable
from time import monotonic
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .ai import (
    AiPipeline,
    AiPipelineError,
    AiResponse,
    AiUnavailableError,
    EmptyUtteranceError,
    NonSpeechUtteranceError,
    build_ai_pipeline,
)
from .config import get_settings
from .protocol import ProtocolError, alert_message, parse_text_message, server_hello, validate_client_hello
from .session import AudioSession, FinalizeReason
from .vad import build_vad_factory


logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])
ai_pipeline: AiPipeline = build_ai_pipeline(get_settings())
vad_factory = build_vad_factory(get_settings())


def _authorized(websocket: WebSocket, token: str) -> bool:
    if not token:
        return True
    expected = token if " " in token else f"Bearer {token}"
    received = websocket.headers.get("authorization", "")
    return secrets.compare_digest(received, expected)


async def ai_readiness() -> dict[str, bool]:
    status = ai_pipeline.health()
    llm = getattr(ai_pipeline, "llm", None)
    ready = getattr(llm, "ready", None)
    if ready is not None:
        status["llm"] = await ready()
    return status


async def _send_ai_result(
    websocket: WebSocket,
    session_id: str,
    result: AiResponse,
    frame_duration_ms: int = 60,
) -> None:
    await websocket.send_json(
        {
            "session_id": session_id,
            "type": "stt",
            "text": result.transcript,
            "language": result.language,
            "language_probability": result.language_probability,
        }
    )
    await websocket.send_json(
        {"session_id": session_id, "type": "llm", "emotion": result.emotion}
    )
    await websocket.send_json(
        {"session_id": session_id, "type": "tts", "state": "start"}
    )
    await websocket.send_json(
        {
            "session_id": session_id,
            "type": "tts",
            "state": "sentence_start",
            "text": result.answer,
        }
    )
    frame_count = len(result.audio_frames)
    expected_playback_seconds = frame_count * frame_duration_ms / 1000
    send_started = monotonic()
    next_deadline = send_started
    frame_interval = frame_duration_ms / 1000
    frames_sent = 0
    last_frame_sent_at = send_started
    try:
        for index, frame in enumerate(result.audio_frames, start=1):
            await websocket.send_bytes(frame)
            frames_sent = index
            last_frame_sent_at = monotonic()
            next_deadline = send_started + index * frame_interval
            await asyncio.sleep(max(0.0, next_deadline - monotonic()))
        actual_send_duration = (
            last_frame_sent_at - send_started if frame_count else 0.0
        )
        await websocket.send_json(
            {"session_id": session_id, "type": "tts", "state": "stop"}
        )
        stop_timestamp = time.time()
        actual_playback_duration = monotonic() - send_started
        logger.info(
            "TTS playback drained: session=%s frames=%d expected_playback_ms=%.1f "
            "actual_send_ms=%.1f actual_playback_ms=%.1f tts_stop_timestamp=%.3f",
            session_id,
            frame_count,
            expected_playback_seconds * 1000,
            actual_send_duration * 1000,
            actual_playback_duration * 1000,
            stop_timestamp,
        )
    except asyncio.CancelledError:
        try:
            await websocket.send_json(
                {"session_id": session_id, "type": "tts", "state": "stop"}
            )
        except (RuntimeError, WebSocketDisconnect):
            pass
        stop_timestamp = time.time()
        logger.info(
            "TTS playback cancelled: session=%s frames_sent=%d frames_total=%d "
            "tts_stop_timestamp=%.3f",
            session_id,
            frames_sent,
            frame_count,
            stop_timestamp,
        )
        raise


async def _close_failed_conversation(
    websocket: WebSocket,
    session_id: str,
    reason: str,
) -> None:
    """Return the device to idle without exposing backend errors to the user.

    The ESP32 firmware moves to ``kDeviceStateIdle`` when its WebSocket audio
    channel closes.  An ``alert`` message only changes the display and leaves
    the device in its current listening state, so it is not suitable for a
    failed AI turn.
    """
    logger.warning(
        "Closing failed AI conversation so the device returns to idle: "
        "session=%s reason=%s",
        session_id,
        reason,
    )
    try:
        await websocket.close(code=1000, reason="AI turn ended")
    except (RuntimeError, WebSocketDisconnect):
        pass


async def _process_utterance(
    websocket: WebSocket,
    session: AudioSession | str,
    frames: list[bytes],
    reason: FinalizeReason,
    resume_after_empty: Callable[[], None] | None = None,
    vad_evidence: dict | None = None,
) -> None:
    if isinstance(session, str):
        session_id = session
        persona = "default"
        history = []
        on_transcript = None
        add_history = None
        customer_profile = None
    else:
        session_id = session.session_id
        persona = session.persona
        history = session.history
        customer_profile = session.customer_profile

        def on_transcript(transcript: str) -> str:
            session.update_persona(transcript)
            # Ollama's deterministic dialogue coordinator updates the profile
            # exactly once. A second update could confirm two pending fields.
            return session.persona

        add_history = session.add_history

    logger.info(
        "Finalizing utterance for session %s: reason=%s frames=%d persona=%s history_turns=%d",
        session_id,
        reason,
        len(frames),
        persona,
        len(history) // 2,
    )
    try:
        kwargs = supported_kwargs(ai_pipeline.run, {"history": history, "persona": persona, "on_transcript": on_transcript, "customer_profile": customer_profile, "vad_evidence": vad_evidence})
        result = await asyncio.wait_for(ai_pipeline.run(frames, **kwargs), timeout=get_settings().ai_pipeline_timeout_seconds)
        logger.info("Conversation session=%s transcript=%r answer=%r confirmed_allergens=%s region=%s resolved_request=%r route=%s response_language=%s sources=%s", session_id, result.transcript, result.answer, customer_profile.allergens if customer_profile else [], customer_profile.region if customer_profile else "", customer_profile.resolved_request if customer_profile else "", customer_profile.last_route if customer_profile else "", result.language, customer_profile.evidence_sources if customer_profile else [])
        if customer_profile is not None:
            logger.info("Dialogue state session=%s action=%s allergy_status=%s pending_allergens=%s pending_note=%r", session_id, customer_profile.dialogue_action, customer_profile.allergy_status, customer_profile.pending_allergens, customer_profile.pending_allergy_note)

        if add_history is not None:
            add_history(result.transcript, result.answer)

        await _send_ai_result(
            websocket,
            session_id,
            result,
            get_settings().frame_duration_ms,
        )
    except asyncio.CancelledError:
        logger.info("AI processing cancelled for session %s", session_id)
        raise
    except AiUnavailableError as exc:
        await _close_failed_conversation(websocket, session_id, str(exc))
    except EmptyUtteranceError:
        logger.info(
            "Empty/noise utterance ignored for session %s; listening ended",
            session_id,
        )
    except NonSpeechUtteranceError:
        logger.info(
            "Combined VAD/ASR non-speech ignored for session %s; listening ended",
            session_id,
        )
    except asyncio.TimeoutError:
        timeout_seconds = get_settings().ai_pipeline_timeout_seconds
        logger.warning(
            "AI pipeline timed out for session %s after %.1fs",
            session_id,
            timeout_seconds,
        )
        await _close_failed_conversation(
            websocket,
            session_id,
            f"AI pipeline timed out after {timeout_seconds:g}s",
        )
    except AiPipelineError as exc:
        logger.warning("AI pipeline error for session %s: %s", session_id, exc)
        await websocket.send_json(alert_message(str(exc), session_id))
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Unexpected AI error for session %s", session_id)
        try:
            await websocket.send_json(alert_message("Local AI processing failed", session_id))
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        frames.clear()


async def _record_vad_rejection(
    session_id: str,
    frames: list[bytes],
    vad_evidence: dict | None = None,
) -> None:
    """Save optional diagnostics, but never call Whisper, Gemma, RAG or TTS."""
    try:
        recorder = getattr(ai_pipeline, "record_rejected_utterance", None)
        if callable(recorder):
            kwargs = supported_kwargs(
                recorder, {"vad_evidence": vad_evidence}
            )
            await recorder(frames, "vad_no_speech", **kwargs)
        logger.info(
            "VAD rejected non-speech before ASR: session=%s frames=%d",
            session_id,
            len(frames),
        )
    except Exception:
        logger.exception("Failed to record VAD-rejected audio for session %s", session_id)
    finally:
        frames.clear()


async def _cancel_inference(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _start_inference(
    websocket: WebSocket,
    session: AudioSession,
    reason: FinalizeReason,
    previous_task: asyncio.Task[None] | None,
) -> asyncio.Task[None] | None:
    await _cancel_inference(previous_task)
    utterance_duration_ms = session.elapsed_seconds * 1000
    frames = session.take_audio()
    vad_evidence = getattr(session, "last_vad_evidence", {"available": False})
    if not frames:
        return None
    if vad_evidence.get("available") and vad_evidence.get("speech_detected") is False:
        async def reject_silence():
            try:
                await _record_vad_rejection(session.session_id, frames, vad_evidence)
            finally:
                # Closing the audio channel returns existing ESP firmware to idle.
                try:
                    await websocket.close(code=1000, reason="No speech detected")
                except (RuntimeError, WebSocketDisconnect):
                    pass
        return asyncio.create_task(reject_silence())
    logger.info(
        "VAD/utterance completed: session=%s reason=%s duration_ms=%.1f",
        session.session_id,
        reason,
        utterance_duration_ms,
    )
    return asyncio.create_task(
        _process_utterance(
            websocket,
            session,
            frames,
            reason,
            vad_evidence=vad_evidence,
        )
    )


@router.websocket("/xiaozhi/v1/")
async def xiaozhi_websocket(websocket: WebSocket) -> None:
    settings = get_settings()
    if not _authorized(websocket, settings.device_token):
        await websocket.close(code=1008, reason="Invalid device token")
        return

    await websocket.accept()
    session = AudioSession(
        session_id=str(uuid4()),
        device_id=websocket.headers.get("device-id", ""),
        client_id=websocket.headers.get("client-id", ""),
        frame_duration_ms=settings.frame_duration_ms,
        max_audio_seconds=settings.max_audio_seconds,
        max_audio_bytes=settings.max_audio_bytes,
        vad_service=vad_factory.create(),
    )
    inference_task: asyncio.Task[None] | None = None

    try:
        first_event = await asyncio.wait_for(
            websocket.receive(), timeout=settings.hello_timeout_seconds
        )
        if first_event["type"] == "websocket.disconnect":
            return
        first_text = first_event.get("text")
        if first_text is None:
            raise ProtocolError("First message must be a JSON hello message")

        first_message = parse_text_message(first_text)
        validate_client_hello(first_message)
        await websocket.send_json(
            server_hello(
                session.session_id,
                settings.server_sample_rate,
                settings.frame_duration_ms,
            )
        )

        while True:
            incoming = await websocket.receive()
            if incoming["type"] == "websocket.disconnect":
                return

            binary = incoming.get("bytes")
            if binary is not None:
                try:
                    reason = session.append_frame(binary)
                except AiPipelineError as exc:
                    session.abort()
                    await websocket.send_json(alert_message(str(exc), session.session_id))
                    continue
                if reason is not None:
                    inference_task = await _start_inference(
                        websocket, session, reason, inference_task
                    )
                continue

            raw_text = incoming.get("text")
            if raw_text is None:
                continue

            message = parse_text_message(raw_text)
            message_type = message["type"]
            if message_type == "listen":
                state = message.get("state")
                if state == "detect":
                    await _cancel_inference(inference_task)
                    inference_task = None
                    session.start("detect")
                elif state == "start":
                    mode = message.get("mode", "auto")
                    if mode not in {"auto", "manual", "realtime"}:
                        raise ProtocolError(f"Unsupported listening mode: {mode}")
                    await _cancel_inference(inference_task)
                    inference_task = None
                    preserve_audio = session.active and session.mode == "detect"
                    session.start(mode, preserve_audio=preserve_audio)
                elif state == "stop":
                    inference_task = await _start_inference(
                        websocket,
                        session,
                        FinalizeReason.MANUAL_STOP,
                        inference_task,
                    )
                else:
                    raise ProtocolError(f"Unsupported listen state: {state}")
            elif message_type == "abort":
                session.abort()
                await _cancel_inference(inference_task)
                inference_task = None
    except TimeoutError:
        await websocket.send_json(alert_message("Timed out waiting for client hello"))
        await websocket.close(code=1008)
    except ProtocolError as exc:
        await websocket.send_json(alert_message(str(exc), session.session_id))
        await websocket.close(code=1003)
    except WebSocketDisconnect:
        return
    finally:
        await _cancel_inference(inference_task)
        session.close()
