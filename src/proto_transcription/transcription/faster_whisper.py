"""Faster Whisper transcriber implementation using CTranslate2.

Uses the faster-whisper library for faster inference than OpenAI's original
Whisper implementation, thanks to CTranslate2 backend.

Reference: https://github.com/guillaumekln/faster-whisper
"""

from pathlib import Path

from faster_whisper import WhisperModel
from loguru import logger

from proto_transcription.exceptions import TranscriptionError
from proto_transcription.transcription.base import BaseTranscriber


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using Faster Whisper with CTranslate2 backend.

    Provides faster inference than OpenAI's Whisper while maintaining quality.
    Models are automatically downloaded on first use.

    Attributes:
        model_size: Size of model (tiny, base, small, medium, large).
    """

    def __init__(self, model_size: str = "base") -> None:
        """Initialize Faster Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
                Defaults to "base" for good speed/accuracy balance.
        """
        super().__init__(model_size)

    @property
    def name(self) -> str:
        """Get transcriber name.

        Returns:
            "faster-whisper"
        """
        return "faster-whisper"

    def load_model(self) -> None:
        """Load Faster Whisper model.

        Downloads model on first use (~140MB for base model).
        Subsequent runs use cached model. Uses CPU with int8 quantization,
        which is optimal for Apple Silicon (CTranslate2 does not support MPS).

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading Faster Whisper {self.model_size} model")
        try:
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("Faster Whisper model loaded successfully")
        except Exception as e:
            raise TranscriptionError(f"Failed to load Faster Whisper model: {e}") from e

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using Faster Whisper.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys

        Raises:
            TranscriptionError: If transcription fails.
        """
        logger.info(f"Transcribing {audio_file.name} with Faster Whisper")
        try:
            segments_gen, info = self.model.transcribe(str(audio_file), beam_size=5)
            logger.debug(f"Detected language: {info.language} ({info.language_probability})")

            segments = [
                {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments_gen
            ]
            formatted = self.format_segments(segments)
            logger.info(f"Transcription complete: {len(segments)} segments")
            return formatted, segments
        except Exception as e:
            raise TranscriptionError(f"Faster Whisper transcription failed: {e}") from e
