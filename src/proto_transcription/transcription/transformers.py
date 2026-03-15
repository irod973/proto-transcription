"""HuggingFace Transformers transcriber implementation.

Uses the HuggingFace Transformers library with the OpenAI Whisper model
from the Hugging Face Model Hub.

Reference: https://huggingface.co/docs/transformers/tasks/asr
"""

from pathlib import Path

import torch
from loguru import logger
from transformers import pipeline

from proto_transcription.exceptions import TranscriptionError
from proto_transcription.transcription.base import BaseTranscriber


class TransformersTranscriber(BaseTranscriber):
    """Transcriber using HuggingFace Transformers and OpenAI Whisper.

    Provides unified interface to Whisper via the popular Transformers library.
    Models are automatically downloaded on first use. Supports MPS acceleration
    on Apple Silicon.

    Attributes:
        model_size: Size of model (tiny, base, small, medium, large).
    """

    def __init__(self, model_size: str = "base") -> None:
        """Initialize Transformers transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
                Defaults to "base" for good speed/accuracy balance.
        """
        super().__init__(model_size)

    @property
    def name(self) -> str:
        """Get transcriber name.

        Returns:
            "transformers"
        """
        return "transformers"

    def load_model(self) -> None:
        """Load Transformers Whisper pipeline.

        Downloads model on first use (~140MB for base model).
        Subsequent runs use cached model. Uses MPS on Apple Silicon,
        falls back to CPU otherwise.

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading Transformers {self.model_size} model")
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.debug(f"Using device: {device}")
            self.model = pipeline(
                task="automatic-speech-recognition",
                model=f"openai/whisper-{self.model_size}",
                device=device,
            )
            logger.info("Transformers model loaded successfully")
        except Exception as e:
            raise TranscriptionError(f"Failed to load Transformers model: {e}") from e

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using Transformers pipeline.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys

        Raises:
            TranscriptionError: If transcription fails.
        """
        logger.info(f"Transcribing {audio_file.name} with Transformers")
        try:
            result = self.model(str(audio_file), return_timestamps=True)
            segments = [
                {
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1],
                    "text": chunk["text"],
                }
                for chunk in result["chunks"]
            ]
            formatted = self.format_segments(segments)
            logger.info(f"Transcription complete: {len(segments)} segments")
            return formatted, segments
        except Exception as e:
            raise TranscriptionError(f"Transformers transcription failed: {e}") from e
