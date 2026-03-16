"""Audio conversion utilities using FFmpeg."""

import shutil
import subprocess  # nosec: B404
from pathlib import Path

from loguru import logger

from proto_transcription.exceptions import AudioConversionError, FFmpegNotFound


class AudioConverter:
    """Convert audio files to WAV format suitable for Whisper.

    Uses FFmpeg to convert various audio formats to 16kHz mono WAV,
    which is recommended input for Whisper models.

    Raises:
        FFmpegNotFound: If ffmpeg is not installed.
    """

    def __init__(self) -> None:
        """Initialize converter and check ffmpeg availability."""
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> None:
        """Check if ffmpeg is installed.

        Raises:
            FFmpegNotFound: If ffmpeg is not found in PATH.
        """
        if shutil.which("ffmpeg") is None:
            raise FFmpegNotFound("ffmpeg is not installed. Install it with: brew install ffmpeg")
        logger.debug("ffmpeg is available")

    @staticmethod
    def is_wav(audio_file: Path) -> bool:
        """Check if audio file is already in WAV format.

        Args:
            audio_file: Path to audio file.

        Returns:
            True if file has .wav extension, False otherwise.
        """
        return audio_file.suffix.lower() == ".wav"

    def convert_to_wav(self, input_file: Path, output_file: Path | None = None) -> Path:
        """Convert audio file to 16kHz mono WAV format.

        If input is already WAV, returns input path unchanged.

        Args:
            input_file: Path to input audio file.
            output_file: Path to output WAV file. If None, uses input filename
                with .wav extension in same directory.

        Returns:
            Path to converted WAV file.

        Raises:
            AudioConversionError: If conversion fails.
        """
        # If already WAV, return input
        if self.is_wav(input_file):
            logger.debug(f"Audio is already WAV: {input_file}")
            return input_file

        # Determine output path
        if output_file is None:
            output_file = input_file.with_suffix(".wav")

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {input_file} to WAV")

        # FFmpeg command: convert to 16kHz mono WAV
        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono (1 channel)
            "-c:a",
            "pcm_s16le",  # 16-bit PCM
            "-y",  # Overwrite output file
            str(output_file),
        ]

        try:
            result = subprocess.run(  # nosec: B603
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg stderr: {result.stderr}")
                raise AudioConversionError(f"ffmpeg conversion failed: {result.stderr}")

            logger.info(f"Successfully converted to: {output_file}")
            return output_file

        except Exception as e:
            if isinstance(e, AudioConversionError):
                raise
            raise AudioConversionError(f"Conversion error: {e}") from e

    def get_duration(self, audio_file: Path) -> str:
        """Get audio file duration.

        Args:
            audio_file: Path to audio file.

        Returns:
            Duration as string in format HH:MM:SS.

        Raises:
            AudioConversionError: If duration query fails.
        """
        cmd = [
            "ffmpeg",
            "-i",
            str(audio_file),
        ]

        try:
            result = subprocess.run(  # nosec: B603
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # FFmpeg outputs duration in stderr
            stderr = result.stderr
            for line in stderr.split("\n"):
                if "Duration:" in line:
                    # Extract duration: "Duration: HH:MM:SS.ms, ..."
                    duration_str = line.split("Duration:")[1].split(",")[0].strip()
                    logger.debug(f"Audio duration: {duration_str}")
                    return duration_str

            raise AudioConversionError("Could not find duration in ffmpeg output")

        except AudioConversionError:
            raise
        except Exception as e:
            raise AudioConversionError(f"Failed to get audio duration: {e}") from e
