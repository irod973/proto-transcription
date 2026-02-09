"""WhisperX transcriber implementation with alignment.

Uses the whisperx library which adds speaker diarization and more accurate
timing alignment compared to base Whisper.

Reference: https://github.com/m-bain/whisperx
"""

from pathlib import Path

from loguru import logger

from proto_transcription.transcription.base import BaseTranscriber


class WhisperXTranscriber(BaseTranscriber):
    """Transcriber using WhisperX with alignment and diarization.

    Provides aligned transcriptions with speaker diarization capabilities.
    Models are automatically downloaded on first use.

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
        Subsequent runs use cached model.

        TODO: Implement model loading
        - Import whisperx
        - Load model with whisperx.load_model()
        - Use self.model_size and device
        - Store in self.model
        - Log successful load

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading WhisperX {self.model_size} model")
        # TODO: Implement
        raise NotImplementedError("WhisperXTranscriber.load_model() not yet implemented")

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using WhisperX.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys

        TODO: Implement transcription
        - Call self.model.transcribe(str(audio_file))
        - Extract and align segments
        - Format with timestamps using self.format_segments()
        - Log and return results

        Raises:
            TranscriptionError: If transcription fails.
        """
        logger.info(f"Transcribing {audio_file.name} with WhisperX")
        # TODO: Implement
        raise NotImplementedError("WhisperXTranscriber.transcribe() not yet implemented")
