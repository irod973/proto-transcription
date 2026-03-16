"""Output file writing for transcriptions."""

from datetime import datetime
from pathlib import Path

from loguru import logger


class TranscriptionWriter:
    """Write transcriptions to files with metadata headers.

    Generates filename based on audio file, model, and model size.
    Includes metadata header with transcription info.
    """

    def __init__(self, output_dir: Path | str) -> None:
        """Initialize writer with output directory.

        Args:
            output_dir: Directory to write transcription files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"TranscriptionWriter initialized with output_dir: {self.output_dir}")

    def generate_filename(self, audio_file: str | Path, model: str, model_size: str) -> str:
        """Generate output filename for transcription.

        Filename format: {audio_basename}_{model}_{model_size}.txt

        Args:
            audio_file: Path or name of audio file.
            model: Model name (e.g., "faster-whisper").
            model_size: Model size (e.g., "base").

        Returns:
            Generated filename as string.
        """
        audio_path = Path(audio_file)
        audio_stem = audio_path.stem
        return f"{audio_stem}_{model}_{model_size}.txt"

    def write_transcription(
        self,
        audio_file: str | Path,
        model: str,
        model_size: str,
        transcription_text: str,
    ) -> Path:
        """Write transcription to file with metadata header.

        Args:
            audio_file: Path or name of audio file.
            model: Model name used for transcription.
            model_size: Model size used for transcription.
            transcription_text: Transcribed text with timestamps.

        Returns:
            Path to written file.
        """
        filename = self.generate_filename(audio_file, model, model_size)
        output_file = self.output_dir / filename

        # Create metadata header
        metadata = self._create_metadata_header(audio_file, model, model_size)

        # Combine metadata and transcription
        full_content = metadata + transcription_text

        # Write to file
        output_file.write_text(full_content)
        logger.info(f"Wrote transcription to: {output_file}")

        return output_file

    @staticmethod
    def _create_metadata_header(audio_file: str | Path, model: str, model_size: str) -> str:
        """Create metadata header for transcription file.

        Args:
            audio_file: Audio file name or path.
            model: Model name.
            model_size: Model size.

        Returns:
            Formatted metadata header as string.
        """
        audio_name = Path(audio_file).name
        timestamp = datetime.now().isoformat()

        header = (
            f"Transcription Metadata\n"
            f"{'=' * 50}\n"
            f"Audio File: {audio_name}\n"
            f"Model: {model}\n"
            f"Model Size: {model_size}\n"
            f"Transcribed: {timestamp}\n"
            f"{'=' * 50}\n\n"
        )

        return header
