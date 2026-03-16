"""Tests for proto_transcription.transcription.base module."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from proto_transcription.transcription.base import BaseTranscriber


class ConcreteTranscriber(BaseTranscriber):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        """Get transcriber name."""
        return "test-transcriber"

    def load_model(self) -> None:
        """Load model."""
        self.model: Any = Mock()

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio file."""
        return "[00:00:00] Test transcription", [
            {"start": 0.0, "end": 1.0, "text": "Test transcription"}
        ]


class TestBaseTranscriber:
    """Tests for BaseTranscriber abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that BaseTranscriber cannot be instantiated directly."""
        # when/then
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTranscriber("base")  # type: ignore[abstract]

    def test_concrete_implementation(self) -> None:
        """Test concrete implementation of BaseTranscriber."""
        # given
        transcriber = ConcreteTranscriber("base")

        # when/then
        assert transcriber.name == "test-transcriber"
        assert transcriber.model_size == "base"

    def test_transcriber_properties(self) -> None:
        """Test transcriber properties."""
        # given
        transcriber = ConcreteTranscriber("small")

        # when/then
        assert transcriber.name == "test-transcriber"
        assert transcriber.model_size == "small"

    def test_load_model(self) -> None:
        """Test loading model."""
        # given
        transcriber = ConcreteTranscriber("base")

        # when
        transcriber.load_model()

        # then
        assert transcriber.model is not None

    def test_transcribe(self, tmp_path: Path) -> None:
        """Test transcription."""
        # given
        transcriber = ConcreteTranscriber("base")
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        # when
        text, segments = transcriber.transcribe(audio_file)

        # then
        assert "[00:00:00]" in text
        assert "Test transcription" in text
        assert len(segments) == 1
        assert segments[0]["text"] == "Test transcription"

    def test_format_timestamp(self) -> None:
        """Test timestamp formatting."""
        # given
        transcriber = ConcreteTranscriber("base")

        # when/then
        assert transcriber.format_timestamp(0.0) == "00:00:00"
        assert transcriber.format_timestamp(65.5) == "00:01:05"
        assert transcriber.format_timestamp(3661.0) == "01:01:01"
        assert transcriber.format_timestamp(36001.0) == "10:00:01"

    def test_format_segments(self) -> None:
        """Test formatting segments with timestamps."""
        # given
        transcriber = ConcreteTranscriber("base")
        segments = [
            {"start": 0.0, "end": 2.0, "text": "First segment"},
            {"start": 2.0, "end": 5.0, "text": "Second segment"},
            {"start": 10.0, "end": 15.0, "text": "Third segment"},
        ]

        # when
        formatted = transcriber.format_segments(segments)

        # then
        lines = formatted.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "[00:00:00] First segment"
        assert lines[1] == "[00:00:02] Second segment"
        assert lines[2] == "[00:00:10] Third segment"

    def test_format_segments_empty(self) -> None:
        """Test formatting empty segments."""
        # given
        transcriber = ConcreteTranscriber("base")

        # when
        formatted = transcriber.format_segments([])

        # then
        assert formatted == ""

    def test_format_segments_with_long_text(self) -> None:
        """Test formatting segments with longer text."""
        # given
        transcriber = ConcreteTranscriber("base")
        segments = [
            {
                "start": 5.5,
                "end": 10.5,
                "text": "This is a longer transcription segment with more words",
            },
        ]

        # when
        formatted = transcriber.format_segments(segments)

        # then
        assert "[00:00:05]" in formatted
        assert "This is a longer transcription segment with more words" in formatted
