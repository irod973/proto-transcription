"""Configuration for the tests."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

# Stub out uninstalled optional heavy dependencies so test collection succeeds
# without requiring the actual libraries to be installed.
_faster_whisper_stub = MagicMock()
sys.modules.setdefault("faster_whisper", _faster_whisper_stub)

_whisperx_stub = MagicMock()
sys.modules.setdefault("whisperx", _whisperx_stub)

from proto_transcription.transcription.base import BaseTranscriber  # noqa: E402


@pytest.fixture
def sample_audio_file(tmp_path: Path) -> Path:
    """Create a minimal valid audio file for testing.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to sample audio file.
    """
    audio_file = tmp_path / "sample.mp3"
    audio_file.touch()
    return audio_file


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for transcriptions.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to temporary output directory.
    """
    output_dir = tmp_path / "transcriptions"
    output_dir.mkdir()
    return output_dir


class MockTranscriber(BaseTranscriber):
    """Mock transcriber for testing."""

    def __init__(self, model_size: str = "base", name_override: str = "mock") -> None:
        """Initialize mock transcriber.

        Args:
            model_size: Model size.
            name_override: Override for transcriber name.
        """
        super().__init__(model_size)
        self._name = name_override

    @property
    def name(self) -> str:
        """Get transcriber name."""
        return self._name

    def load_model(self) -> None:
        """Load model (mocked)."""
        self.model: Any = Mock()

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio (mocked).

        Args:
            audio_file: Path to audio file.

        Returns:
            Mock transcription with timestamps.
        """
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"},
        ]
        formatted = self.format_segments(segments)
        return formatted, segments


@pytest.fixture
def mock_transcriber(tmp_path: Path) -> MockTranscriber:
    """Create a mock transcriber for testing.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        MockTranscriber instance.
    """
    return MockTranscriber("base")
