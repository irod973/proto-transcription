"""CLI entry point for proto-transcription."""

from pathlib import Path
from typing import Type, cast

import click
from loguru import logger

from proto_transcription.audio.converter import AudioConverter
from proto_transcription.config import TranscriptionConfig
from proto_transcription.exceptions import (
    FFmpegNotFound,
    InvalidAudioFile,
    ProtoTranscriptionException,
)
from proto_transcription.output.writer import TranscriptionWriter
from proto_transcription.transcription.base import BaseTranscriber
from proto_transcription.transcription.faster_whisper import FasterWhisperTranscriber
from proto_transcription.transcription.transformers import TransformersTranscriber
from proto_transcription.transcription.whisperx import WhisperXTranscriber


@click.group()
def cli() -> None:
    """Transcribe audio files using Whisper models.

    Supports three Whisper implementations for comparison:
    - Faster Whisper (CTranslate2 backend)
    - WhisperX (with alignment)
    - HuggingFace Transformers

    All transcriptions include timestamps for easy quote extraction.
    """
    pass


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./transcriptions",
    help="Output directory for transcriptions",
)
@click.option(
    "--models",
    "-m",
    type=str,
    default="faster-whisper,whisperx,transformers",
    help="Comma-separated model names (faster-whisper, whisperx, transformers)",
)
@click.option(
    "--model-size",
    "-s",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    default="base",
    help="Model size for all transcribers",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def transcribe(
    audio_file: str,
    output_dir: str,
    models: str,
    model_size: str,
    verbose: bool,
) -> None:
    """Transcribe AUDIO_FILE using selected Whisper models.

    Outputs separate files for each model with timestamps:
    {audio}_{model}_{size}.txt

    Example:
        proto-transcription transcribe podcast.mp3 --output-dir ./results
    """
    try:
        # Setup logging
        if verbose:
            logger.enable("proto_transcription")
            logger.info("Verbose logging enabled")
        else:
            # Disable debug messages in normal mode
            logger.disable("proto_transcription")

        # Parse models
        model_list = tuple(m.strip() for m in models.split(","))

        # Validate configuration
        config = TranscriptionConfig(
            audio_file=audio_file,
            output_dir=output_dir,
            models=model_list,
            model_size=model_size,
            verbose=verbose,
        )
        logger.info("Configuration validated")

        # Cast audio_file to Path (guaranteed by __post_init__)
        audio_path = cast(Path, config.audio_file)
        output_path = cast(Path, config.output_dir)

        # Check ffmpeg availability
        try:
            audio_converter = AudioConverter()
        except FFmpegNotFound:
            click.echo("Error: ffmpeg is not installed", err=True)
            click.echo("Install with: brew install ffmpeg", err=True)
            raise SystemExit(1)

        # Convert audio to WAV
        wav_file = audio_converter.convert_to_wav(audio_path)
        logger.info(f"Audio file ready: {wav_file}")

        # Get duration
        try:
            duration = audio_converter.get_duration(audio_path)
            click.echo(f"Audio duration: {duration}")
        except Exception as e:
            logger.warning(f"Could not get duration: {e}")

        # Initialize output writer
        writer = TranscriptionWriter(output_path)
        logger.info(f"Output directory: {output_path}")

        # Initialize transcribers
        transcriber_classes: dict[str, Type[BaseTranscriber]] = {
            "faster-whisper": FasterWhisperTranscriber,
            "whisperx": WhisperXTranscriber,
            "transformers": TransformersTranscriber,
        }

        transcribers: list[BaseTranscriber] = []
        for model_name in config.models:
            if model_name not in transcriber_classes:
                click.echo(f"Error: Unknown model {model_name}", err=True)
                raise SystemExit(1)

            transcriber_class = transcriber_classes[model_name]
            transcriber = transcriber_class(model_size)
            transcribers.append(transcriber)
            logger.info(f"Initialized {model_name} transcriber")

        # Run transcription with each model
        click.echo(f"Transcribing with {len(transcribers)} models...")

        for transcriber in transcribers:
            try:
                click.echo(f"  Loading {transcriber.name}...", nl=False)

                # Load model
                transcriber.load_model()
                click.echo(" done")

                # Transcribe
                click.echo(f"  Transcribing with {transcriber.name}...", nl=False)
                text, segments = transcriber.transcribe(wav_file)
                click.echo(" done")

                # Write output
                output_file = writer.write_transcription(
                    audio_path.name,
                    transcriber.name,
                    config.model_size,
                    text,
                )

                click.echo(f"    → {output_file.name}")
                logger.info(f"Transcription saved: {output_file}")

            except NotImplementedError:
                click.echo(" [NOT IMPLEMENTED]", err=True)
                logger.error(f"{transcriber.name} transcriber not yet implemented")
            except Exception as e:
                click.echo(" [ERROR]", err=True)
                logger.error(f"Transcription failed: {e}")
                click.echo(f"Error: {e}", err=True)

        click.echo(f"✓ Transcription complete. Results in: {output_path}")

    except InvalidAudioFile as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except ProtoTranscriptionException as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        click.echo(f"Unexpected error: {e}", err=True)
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Command line arguments (for testing).

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    try:
        cli(argv, standalone_mode=False)
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception:
        return 1


if __name__ == "__main__":
    cli()
