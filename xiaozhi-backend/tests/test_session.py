from app.session import AudioSession, FinalizeReason
from app.vad import TimeoutVadService


class Clock:
    value = 0.0

    def __call__(self) -> float:
        return self.value


def make_session(clock: Clock, max_seconds: int = 30, max_bytes: int = 100) -> AudioSession:
    return AudioSession(
        session_id="session",
        device_id="device",
        client_id="client",
        frame_duration_ms=60,
        max_audio_seconds=max_seconds,
        max_audio_bytes=max_bytes,
        vad_service=TimeoutVadService(8),
        clock=clock,
    )


def test_abort_clears_audio() -> None:
    session = make_session(Clock())
    session.start("manual")
    session.append_frame(b"audio")
    session.abort()

    assert session.frame_count == 0
    assert session.total_bytes == 0
    assert session.active is False


def test_disconnect_cleanup_clears_audio() -> None:
    session = make_session(Clock())
    session.start("manual")
    session.append_frame(b"audio")
    session.clear()

    assert session.frame_count == 0
    assert session.total_bytes == 0


def test_hard_buffer_limit() -> None:
    session = make_session(Clock(), max_bytes=4)
    session.start("manual")
    session.append_frame(b"1234")

    reason = session.append_frame(b"5")

    assert reason == FinalizeReason.BUFFER_LIMIT
    assert session.total_bytes == 4


def test_timeout_fallback_is_not_vad() -> None:
    clock = Clock()
    session = make_session(clock)
    session.start("auto")
    session.append_frame(b"audio")
    clock.value = 8.0

    reason = session.append_frame(b"audio-2")

    assert reason == FinalizeReason.AUTO_TIMEOUT


def test_manual_mode_still_feeds_vad_for_speech_evidence() -> None:
    class EvidenceVad:
        speech_detected = False

        def should_finalize(self, frame, elapsed_seconds):
            del frame, elapsed_seconds
            self.speech_detected = True
            return True

        def reset(self):
            pass

        def close(self):
            pass

    clock = Clock()
    session = AudioSession(
        "session", "device", "client", 60, 30, 100,
        EvidenceVad(), clock,
    )
    session.start("manual")

    assert session.append_frame(b"audio") is None
    assert session.vad_service.speech_detected is True


def test_culinary_persona_switching() -> None:
    session = make_session(Clock())
    assert session.persona == "default"

    # Culinary personas switch correctly
    assert session.update_persona("bạn bây giờ là bạn thân nhé") is True
    assert session.persona == "friend"

    assert session.update_persona("hãy đóng vai đầu bếp") is True
    assert session.persona == "chef"

    assert session.update_persona("hãy làm chuyên gia dinh dưỡng cho tôi") is True
    assert session.persona == "nutritionist"

    assert session.update_persona("quay lại bình thường") is True
    assert session.persona == "default"

    # Non-food roles are rejected and keep current persona
    assert session.update_persona("bạn hãy làm giáo viên dạy toán") is False
    assert session.persona == "default"

    assert session.update_persona("hãy đóng vai phi hành gia đi") is False
    assert session.persona == "default"

    assert session.update_persona("đóng vai cảnh sát bắt cướp") is False
    assert session.persona == "default"


def test_culinary_history_cleared_on_topic_change() -> None:
    session = make_session(Clock())
    session.update_customer_profile("Tôi dị ứng đậu phộng")
    session.update_customer_profile("Đúng")
    session.add_history("hôm nay ăn gì ngon", "Bạn có thể thử món bún chả Hà Nội")
    assert len(session.history) == 2

    assert session.update_persona("đổi chủ đề khác đi") is True
    assert len(session.history) == 0
    assert session.customer_profile.allergens == ["đậu phộng"]


def test_full_reset_requires_allergy_onboarding_again() -> None:
    session = make_session(Clock())
    session.update_customer_profile("Tôi dị ứng tôm")
    session.update_customer_profile("Đúng")

    assert session.update_persona("bắt đầu lại") is True
    assert session.customer_profile.allergy_status == "unknown"
    assert session.customer_profile.allergens == []
