"""Configuration for proto_transcription transcription workflow."""

from dataclasses import dataclass
from pathlib import Path

from proto_transcription.exceptions import InvalidAudioFile, InvalidModelName

VALID_MODELS = frozenset(("faster-whisper", "whisperx", "transformers"))
VALID_MODEL_SIZES = frozenset(("tiny", "base", "small", "medium", "large"))


@dataclass
class TranscriptionConfig:
    """Configuration for transcription workflow.

    Args:
        audio_file: Path to audio file (MP3, WAV, etc.)
        output_dir: Directory to write transcription files
        models: Tuple of model names to use
        model_size: Model size (tiny, base, small, medium, large)
        verbose: Enable verbose logging

    Raises:
        InvalidAudioFile: If audio file does not exist
        InvalidModelName: If model name is invalid
        ValueError: If model size is invalid
    """

    audio_file: str | Path
    output_dir: str | Path
    models: tuple[str, ...] = ("faster-whisper", "whisperx", "transformers")
    model_size: str = "base"
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration and convert paths."""
        # Convert to Path objects
        self.audio_file = Path(self.audio_file)
        self.output_dir = Path(self.output_dir)

        # Validate audio file exists
        if not self.audio_file.exists():
            raise InvalidAudioFile(f"Audio file not found: {self.audio_file}")

        # Validate model names
        for model in self.models:
            if model not in VALID_MODELS:
                raise InvalidModelName(
                    f"Invalid model name: {model}. Valid models: {', '.join(VALID_MODELS)}"
                )

        # Validate model size
        if self.model_size not in VALID_MODEL_SIZES:
            raise ValueError(
                f"Invalid model size: {self.model_size}. "
                f"Valid sizes: {', '.join(VALID_MODEL_SIZES)}"
            )

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
