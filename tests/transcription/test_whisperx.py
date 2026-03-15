"""Tests for WhisperXTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_transcription.transcription.whisperx import WhisperXTranscriber


class TestWhisperXTranscriber:
    """Tests for WhisperXTranscriber."""

    def test_name(self) -> None:
        """Test transcriber name."""
        transcriber = WhisperXTranscriber("base")
        assert transcriber.name == "whisperx"

    def test_load_model(self) -> None:
        """Test that load_model calls whisperx.load_model and stores result."""
        transcriber = WhisperXTranscriber("base")
        mock_model = MagicMock()

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_wx.load_model.return_value = mock_model
            transcriber.load_model()

            mock_wx.load_model.assert_called_once_with("base", device="cpu", compute_type="int8")
            assert transcriber.model is mock_model

    def test_transcribe_returns_formatted_text_and_segments(self, tmp_path: Path) -> None:
        """Test transcribe returns aligned, formatted segments."""
        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        raw_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.0, "end": 4.5, "text": "How are you"},
        ]
        aligned_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.0, "end": 4.5, "text": "How are you"},
        ]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": raw_segments, "language": "en"}
        transcriber.model = mock_model

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_align_model = MagicMock()
            mock_metadata = MagicMock()
            mock_wx.load_align_model.return_value = (mock_align_model, mock_metadata)
            mock_wx.align.return_value = {"segments": aligned_segments}
            mock_wx.load_audio.return_value = MagicMock()

            text, segments = transcriber.transcribe(audio_file)

        assert "[00:00:00]" in text
        assert "Hello world" in text
        assert "[00:00:02]" in text
        assert "How are you" in text

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 2.0, "text": "Hello world"}

    def test_transcribe_uses_aligned_result(self, tmp_path: Path) -> None:
        """Test that segments come from the aligned result, not original transcription."""
        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        raw_segments = [{"start": 0.0, "end": 5.0, "text": "Original"}]
        # Alignment refines timestamps
        aligned_segments = [{"start": 0.1, "end": 4.9, "text": "Original"}]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": raw_segments, "language": "en"}
        transcriber.model = mock_model

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
            mock_wx.align.return_value = {"segments": aligned_segments}
            mock_wx.load_audio.return_value = MagicMock()

            _, segments = transcriber.transcribe(audio_file)

        # Should use aligned timestamps (0.1), not raw (0.0)
        assert segments[0]["start"] == 0.1

    def test_transcribe_wraps_exception_in_transcription_error(self, tmp_path: Path) -> None:
        """Test that library exceptions are wrapped in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("model crashed")
        transcriber.model = mock_model

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_wx.load_audio.return_value = MagicMock()
            with pytest.raises(TranscriptionError, match="model crashed"):
                transcriber.transcribe(audio_file)

    def test_load_model_wraps_exception_in_transcription_error(self) -> None:
        """Test that load_model wraps library exceptions in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = WhisperXTranscriber("base")

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_wx.load_model.side_effect = RuntimeError("download failed")
            with pytest.raises(TranscriptionError, match="download failed"):
                transcriber.load_model()

    def test_transcribe_calls_model_with_correct_args(self, tmp_path: Path) -> None:
        """Test transcribe calls model.transcribe with audio and batch_size."""
        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_audio = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "en"}
        transcriber.model = mock_model

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_wx.load_audio.return_value = mock_audio
            mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
            mock_wx.align.return_value = {"segments": []}

            transcriber.transcribe(audio_file)

        mock_model.transcribe.assert_called_once_with(mock_audio, batch_size=16)

    def test_transcribe_raises_if_model_not_loaded(self, tmp_path: Path) -> None:
        """Test transcribe raises TranscriptionError if load_model() was not called."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        with pytest.raises(TranscriptionError, match="Model not loaded"):
            transcriber.transcribe(audio_file)

    def test_transcribe_calls_alignment_with_correct_args(self, tmp_path: Path) -> None:
        """Test alignment calls use correct device and return_char_alignments=False."""
        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_audio = MagicMock()
        mock_model = MagicMock()
        mock_segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        mock_model.transcribe.return_value = {"segments": mock_segments, "language": "en"}
        transcriber.model = mock_model

        mock_align_model = MagicMock()
        mock_metadata = MagicMock()

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_wx.load_audio.return_value = mock_audio
            mock_wx.load_align_model.return_value = (mock_align_model, mock_metadata)
            mock_wx.align.return_value = {"segments": mock_segments}

            transcriber.transcribe(audio_file)

        mock_wx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
        mock_wx.align.assert_called_once_with(
            mock_segments,
            mock_align_model,
            mock_metadata,
            mock_audio,
            "cpu",
            return_char_alignments=False,
        )
