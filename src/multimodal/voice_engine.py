from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VoiceEngineState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class VoiceConfig:
    stt_model: str = "base"
    tts_rate: int = 175
    tts_volume: float = 1.0
    timeout_s: float = 5.0
    phrase_limit_s: float = 30.0


class VoiceEngine:
    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()
        self._state = VoiceEngineState.IDLE

    @property
    def state(self) -> VoiceEngineState:
        return self._state

    def transcribe(self, audio_data: bytes, format: str = "wav") -> str:
        self._state = VoiceEngineState.PROCESSING
        try:
            result = self._transcribe_internal(audio_data, format)
            return result
        finally:
            self._state = VoiceEngineState.IDLE

    def synthesize(self, text: str, voice: str = "default") -> bytes:
        self._state = VoiceEngineState.SPEAKING
        try:
            result = self._synthesize_internal(text, voice)
            return result
        finally:
            self._state = VoiceEngineState.IDLE

    def listen(self, timeout: float | None = None) -> bytes:
        self._state = VoiceEngineState.LISTENING
        try:
            result = self._listen_internal(timeout or self.config.timeout_s)
            return result
        finally:
            self._state = VoiceEngineState.IDLE

    def available_voices(self) -> list[str]:
        voices: list[str] = ["default"]
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split("\n"):
                parts = line.split()
                if parts:
                    voices.append(parts[0].rstrip())
        except Exception:
            pass
        return voices

    def _transcribe_internal(self, audio_data: bytes, format: str) -> str:
        try:
            import speech_recognition as sr
        except ImportError:
            return "[transcription unavailable - speech_recognition not installed]"
        recognizer = sr.Recognizer()
        audio_file = io.BytesIO(audio_data)
        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
        except Exception:
            return ""

    def _synthesize_internal(self, text: str, voice: str) -> bytes:
        if os.name == "posix":
            result = subprocess.run(
                ["say", "-v", voice if voice != "default" else "Samantha", text],
                capture_output=True,
                timeout=30,
            )
            return result.stdout or text.encode("utf-8")
        return text.encode("utf-8")

    def _listen_internal(self, timeout: float) -> bytes:
        try:
            import speech_recognition as sr
        except ImportError:
            return b"[listening unavailable]"
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=self.config.phrase_limit_s)
                return audio.get_wav_data()
        except Exception:
            return b""
