import sys
import struct
import wave
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice not installed. Run: pip install sounddevice")

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
WAV_PATH = "test_mic_recording.wav"

def run():
    print("\n=== Available Microphones ===")
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- DEFAULT" if i == default_in else ""
            print(f"  [{i}] {dev['name']}{marker}")

    print(f"\nRecording {RECORD_SECONDS}s from default mic — say something!\n")

    audio_chunks = []

    def callback(indata, frames, time_info, status):
        audio_chunks.append(indata.copy())
        chunk_max = int(np.abs(indata).max() * 32767)
        bar = "#" * min(30, chunk_max // 1000)
        print(f"\r  Level: [{bar:<30}] {chunk_max:5d}", end="", flush=True)

    with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="int16", callback=callback):
        sd.sleep(RECORD_SECONDS * 1000)

    print()

    audio = np.concatenate(audio_chunks, axis=0)
    max_amp = int(np.abs(audio).max())
    print(f"\nMax amplitude: {max_amp}")

    if max_amp < 500:
        print("WARNING: Very low amplitude! Mic may not be detecting your voice.")
        print("  Check: Windows Sound Settings > Input > your mic volume level.")
    else:
        print("OK: Mic is working.")

    with wave.open(WAV_PATH, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())
    print(f"Saved to: {WAV_PATH}")

    if HAS_WHISPER:
        print("\nTranscribing with faster-whisper (model=small)...")
        model = WhisperModel("small", device="cpu", compute_type="int8", local_files_only=True)
        segments, info = model.transcribe(WAV_PATH, beam_size=5)
        segments = list(segments)
        print(f"Language: {info.language} ({info.language_probability:.2f})")
        if segments:
            for s in segments:
                print(f"  [{s.start:.1f}s->{s.end:.1f}s] {s.text.strip()}")
        else:
            print("  (no speech detected)")
    else:
        print("\nfaster-whisper not found - skipping transcription.")
        print("You can listen to test_mic_recording.wav manually to verify mic.")

run()
