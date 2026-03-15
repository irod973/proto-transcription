"""Tests for TransformersTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_transcription.transcription.transformers import TransformersTranscriber


class TestTransformersTranscriber:
    """Tests for TransformersTranscriber."""

    def test_name(self) -> None:
        """Test transcriber name."""
        transcriber = TransformersTranscriber("base")
        assert transcriber.name == "transformers"

    def test_load_model_uses_mps_when_available(self) -> None:
        """Test load_model uses MPS device when available."""
        transcriber = TransformersTranscriber("base")
        mock_pipe = MagicMock()

        with (
            patch("proto_transcription.transcription.transformers.torch") as mock_torch,
            patch(
                "proto_transcription.transcription.transformers.pipeline",
                return_value=mock_pipe,
            ) as mock_pipeline,
        ):
            mock_torch.backends.mps.is_available.return_value = True

            transcriber.load_model()

            mock_pipeline.assert_called_once_with(
                task="automatic-speech-recognition",
                model="openai/whisper-base",
                device="mps",
            )
            assert transcriber.model is mock_pipe

    def test_load_model_falls_back_to_cpu(self) -> None:
        """Test load_model falls back to CPU when MPS is unavailable."""
        transcriber = TransformersTranscriber("small")
        mock_pipe = MagicMock()

        with (
            patch("proto_transcription.transcription.transformers.torch") as mock_torch,
            patch(
                "proto_transcription.transcription.transformers.pipeline",
                return_value=mock_pipe,
            ) as mock_pipeline,
        ):
            mock_torch.backends.mps.is_available.return_value = False

            transcriber.load_model()

            mock_pipeline.assert_called_once_with(
                task="automatic-speech-recognition",
                model="openai/whisper-small",
                device="cpu",
            )
            assert transcriber.model is mock_pipe

    def test_transcribe_returns_formatted_text_and_segments(self, tmp_path: Path) -> None:
        """Test transcribe parses HuggingFace chunks into standard segment format."""
        transcriber = TransformersTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        # HuggingFace pipeline returns chunks with (start, end) timestamp tuples
        pipeline_result = {
            "text": "Hello world. How are you.",
            "chunks": [
                {"text": " Hello world.", "timestamp": (0.0, 2.5)},
                {"text": " How are you.", "timestamp": (2.5, 5.0)},
            ],
        }
        mock_pipe = MagicMock(return_value=pipeline_result)
        transcriber.model = mock_pipe

        text, segments = transcriber.transcribe(audio_file)

        assert "[00:00:00]" in text
        assert "Hello world." in text
        assert "[00:00:02]" in text
        assert "How are you." in text

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 2.5, "text": " Hello world."}
        assert segments[1] == {"start": 2.5, "end": 5.0, "text": " How are you."}

    def test_transcribe_calls_pipeline_with_timestamps(self, tmp_path: Path) -> None:
        """Test transcribe calls pipeline with return_timestamps=True."""
        transcriber = TransformersTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_pipe = MagicMock(return_value={"text": "", "chunks": []})
        transcriber.model = mock_pipe

        transcriber.transcribe(audio_file)

        mock_pipe.assert_called_once_with(str(audio_file), return_timestamps=True)

    def test_transcribe_wraps_exception_in_transcription_error(self, tmp_path: Path) -> None:
        """Test that library exceptions are wrapped in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = TransformersTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_pipe = MagicMock(side_effect=RuntimeError("pipeline failed"))
        transcriber.model = mock_pipe

        with pytest.raises(TranscriptionError, match="pipeline failed"):
            transcriber.transcribe(audio_file)

    def test_load_model_wraps_exception_in_transcription_error(self) -> None:
        """Test that load_model wraps library exceptions in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = TransformersTranscriber("base")

        with (
            patch("proto_transcription.transcription.transformers.torch") as mock_torch,
            patch(
                "proto_transcription.transcription.transformers.pipeline",
                side_effect=RuntimeError("download failed"),
            ),
        ):
            mock_torch.backends.mps.is_available.return_value = False
            with pytest.raises(TranscriptionError, match="download failed"):
                transcriber.load_model()

    def test_transcribe_raises_if_model_not_loaded(self, tmp_path: Path) -> None:
        """Test transcribe raises TranscriptionError if load_model() was not called."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = TransformersTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        with pytest.raises(TranscriptionError, match="Model not loaded"):
            transcriber.transcribe(audio_file)
