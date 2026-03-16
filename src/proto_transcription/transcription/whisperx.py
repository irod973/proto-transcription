"""WhisperX transcriber implementation with alignment.

Uses the whisperx library which adds more accurate word-level timing alignment
compared to base Whisper.

Reference: https://github.com/m-bain/whisperx
"""

from pathlib import Path

import whisperx
from loguru import logger

from proto_transcription.exceptions import TranscriptionError
from proto_transcription.transcription.base import BaseTranscriber


class WhisperXTranscriber(BaseTranscriber):
    """Transcriber using WhisperX with alignment.

    Provides aligned transcriptions with accurate word-level timestamps.
    Models are automatically downloaded on first use. Uses CPU (MPS support
    in WhisperX is limited).

    Attributes:
        model_size: Size of model (tiny, base, small, medium, large).
    """

    def __init__(self, model_size: str = "base") -> None:
        """Initialize WhisperX transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
                Defaults to "base" for good speed/accuracy balance.
        """
        super().__init__(model_size)

    @property
    def name(self) -> str:
        """Get transcriber name.

        Returns:
            "whisperx"
        """
        return "whisperx"

    def load_model(self) -> None:
        """Load WhisperX model.

        Downloads model on first use (~140MB for base model).
        Subsequent runs use cached model. Uses CPU with int8 quantization
        (WhisperX has limited MPS support).

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading WhisperX {self.model_size} model")
        try:
            self.model = whisperx.load_model(self.model_size, device="cpu", compute_type="int8")
            logger.info("WhisperX model loaded successfully")
        except Exception as e:
            raise TranscriptionError(f"Failed to load WhisperX model: {e}") from e

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using WhisperX with alignment.

        Runs transcription then applies forced phoneme alignment for more
        accurate word-level timestamps.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys (from aligned result)

        Raises:
            TranscriptionError: If transcription or alignment fails.
        """
        if self.model is None:
            raise TranscriptionError("Model not loaded. Call load_model() first.")
        logger.info(f"Transcribing {audio_file.name} with WhisperX")
        try:
            audio = whisperx.load_audio(str(audio_file))
            result = self.model.transcribe(audio, batch_size=16)
            logger.debug(f"Detected language: {result['language']}")

            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device="cpu"
            )
            aligned = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                "cpu",
                return_char_alignments=False,
            )

            segments = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                for seg in aligned["segments"]
            ]
            formatted = self.format_segments(segments)
            logger.info(f"Transcription complete: {len(segments)} segments")
            return formatted, segments
        except Exception as e:
            raise TranscriptionError(f"WhisperX transcription failed: {e}") from e
