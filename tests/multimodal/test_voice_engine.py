import shutil

import pytest

from src.multimodal.voice_engine import VoiceConfig, VoiceEngine, VoiceEngineState


class TestVoiceEngine:
    def test_initial_state(self):
        ve = VoiceEngine()
        assert ve.state is VoiceEngineState.IDLE

    def test_transcribe_no_speech_recognition(self):
        ve = VoiceEngine()
        result = ve.transcribe(b"fake audio data")
        assert "unavailable" in result

    @pytest.mark.skipif(
        not shutil.which("say"),
        reason="requires macOS 'say' command",
    )
    def test_synthesize_returns_bytes(self):
        ve = VoiceEngine()
        result = ve._synthesize_internal("hello", "default")
        assert isinstance(result, bytes)

    def test_listen_no_microphone(self):
        ve = VoiceEngine()
        result = ve.listen(timeout=0.01)
        assert isinstance(result, bytes)

    def test_available_voices_includes_default(self):
        ve = VoiceEngine()
        voices = ve.available_voices()
        assert "default" in voices

    def test_config_custom(self):
        config = VoiceConfig(tts_rate=200, phrase_limit_s=15.0)
        ve = VoiceEngine(config)
        assert ve.config.tts_rate == 200
        assert ve.config.phrase_limit_s == 15.0

    def test_state_machine(self):
        ve = VoiceEngine()
        assert ve.state is VoiceEngineState.IDLE
        ve.synthesize("test")
        assert ve.state is VoiceEngineState.IDLE
