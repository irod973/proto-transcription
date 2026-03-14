"""Tests for FasterWhisperTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_transcription.transcription.faster_whisper import FasterWhisperTranscriber


class TestFasterWhisperTranscriber:
    """Tests for FasterWhisperTranscriber."""

    def test_name(self) -> None:
        """Test transcriber name."""
        transcriber = FasterWhisperTranscriber("base")
        assert transcriber.name == "faster-whisper"

    def test_load_model(self) -> None:
        """Test that load_model instantiates WhisperModel and stores it."""
        transcriber = FasterWhisperTranscriber("base")
        mock_model = MagicMock()

        with patch(
            "proto_transcription.transcription.faster_whisper.WhisperModel",
            return_value=mock_model,
        ) as mock_cls:
            transcriber.load_model()

            mock_cls.assert_called_once_with("base", device="cpu", compute_type="int8")
            assert transcriber.model is mock_model

    def test_transcribe_returns_formatted_text_and_segments(self, tmp_path: Path) -> None:
        """Test transcribe returns formatted text with timestamps and segment dicts."""
        transcriber = FasterWhisperTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        # Mock segments as named-tuple-like objects (faster-whisper returns these)
        seg1 = MagicMock()
        seg1.start = 0.0
        seg1.end = 2.5
        seg1.text = " Hello world"

        seg2 = MagicMock()
        seg2.start = 2.5
        seg2.end = 5.0
        seg2.text = " Goodbye world"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), MagicMock())
        transcriber.model = mock_model

        text, segments = transcriber.transcribe(audio_file)

        assert "[00:00:00]" in text
        assert "Hello world" in text
        assert "[00:00:02]" in text
        assert "Goodbye world" in text

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 2.5, "text": " Hello world"}
        assert segments[1] == {"start": 2.5, "end": 5.0, "text": " Goodbye world"}

    def test_transcribe_calls_model_with_correct_args(self, tmp_path: Path) -> None:
        """Test transcribe calls model.transcribe with str path and beam_size."""
        transcriber = FasterWhisperTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock())
        transcriber.model = mock_model

        transcriber.transcribe(audio_file)

        mock_model.transcribe.assert_called_once_with(str(audio_file), beam_size=5)

    def test_load_model_wraps_exception_in_transcription_error(self) -> None:
        """Test that load_model wraps library exceptions in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = FasterWhisperTranscriber("base")

        with patch(
            "proto_transcription.transcription.faster_whisper.WhisperModel",
            side_effect=RuntimeError("download failed"),
        ):
            with pytest.raises(TranscriptionError, match="download failed"):
                transcriber.load_model()

    def test_transcribe_wraps_exception_in_transcription_error(self, tmp_path: Path) -> None:
        """Test that library exceptions are wrapped in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = FasterWhisperTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("model failed")
        transcriber.model = mock_model

        with pytest.raises(TranscriptionError, match="model failed"):
            transcriber.transcribe(audio_file)
