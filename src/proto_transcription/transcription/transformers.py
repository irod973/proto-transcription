"""HuggingFace Transformers transcriber implementation.

Uses the HuggingFace Transformers library with the OpenAI Whisper model
from the Hugging Face Model Hub.

Reference: https://huggingface.co/docs/transformers/tasks/asr
"""

from pathlib import Path

from loguru import logger

from proto_transcription.transcription.base import BaseTranscriber


class TransformersTranscriber(BaseTranscriber):
    """Transcriber using HuggingFace Transformers and OpenAI Whisper.

    Provides unified interface to Whisper via the popular Transformers library.
    Models are automatically downloaded on first use.

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
        """Load Transformers Whisper model.

        Downloads model on first use (~140MB for base model).
        Subsequent runs use cached model.

        TODO: Implement model loading
        - Import from transformers: pipeline, AutoModelForSpeechSeq2Seq
        - Create or load processor and model
        - Set device (cuda if available, else cpu)
        - Create pipeline with model and processor
        - Store in self.model
        - Log successful load

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading Transformers {self.model_size} model")
        # TODO: Implement
        raise NotImplementedError("TransformersTranscriber.load_model() not yet implemented")

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using Transformers pipeline.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys

        TODO: Implement transcription
        - Call self.model(str(audio_file))
        - Extract segments from result["chunks"]
        - Format with timestamps using self.format_segments()
        - Log and return results

        Note: May need to extract timing info from model output.
        Transformers pipeline may provide different segment format.

        Raises:
            TranscriptionError: If transcription fails.
        """
        logger.info(f"Transcribing {audio_file.name} with Transformers")
        # TODO: Implement
        raise NotImplementedError("TransformersTranscriber.transcribe() not yet implemented")
