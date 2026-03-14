"""Abstract base class for Whisper transcriber implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTranscriber(ABC):
    """Abstract base class for Whisper transcriber implementations.

    All transcriber implementations must inherit from this class and implement
    the required abstract methods.

    Attributes:
        model_size: Size of model to use (tiny, base, small, medium, large).
        model: Loaded model object (set by load_model()).
    """

    def __init__(self, model_size: str) -> None:
        """Initialize transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
        """
        self.model_size = model_size
        self.model: Any = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Get transcriber name.

        Returns:
            Name of transcriber (e.g., "faster-whisper", "whisperx").
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the transcriber model.

        Loads the model and sets self.model. May download large model files
        on first use (~140MB for base models).

        Raises:
            TranscriptionError: If model loading fails.
        """
        pass

    @abstractmethod
    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio file.

        Args:
            audio_file: Path to audio file (WAV, MP3, etc).

        Returns:
            Tuple of (transcribed_text, segments) where:
            - transcribed_text: Full transcription formatted with timestamps
            - segments: List of dicts with keys: start, end, text

        Raises:
            TranscriptionError: If transcription fails.
        """
        pass

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS timestamp.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted timestamp string.

        Examples:
            >>> BaseTranscriber.format_timestamp(0.0)
            '00:00:00'
            >>> BaseTranscriber.format_timestamp(65.5)
            '00:01:05'
            >>> BaseTranscriber.format_timestamp(3661.0)
            '01:01:01'
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def format_segments(self, segments: list[dict]) -> str:
        """Format transcription segments with timestamps.

        Args:
            segments: List of segment dicts with start, end, text keys.

        Returns:
            Formatted transcription with timestamps, one per line.
            Format: [HH:MM:SS] transcribed text

        Example:
            >>> segments = [
            ...     {"start": 0.0, "end": 2.0, "text": "Hello world"},
            ...     {"start": 2.0, "end": 5.0, "text": "This is a test"},
            ... ]
            >>> transcriber.format_segments(segments)
            '[00:00:00] Hello world\\n[00:00:02] This is a test'
        """
        if not segments:
            return ""

        lines = []
        for segment in segments:
            timestamp = self.format_timestamp(segment["start"])
            text = segment["text"]
            lines.append(f"[{timestamp}] {text}")

        return "\n".join(lines)
