"""Tests for proto_transcription.config module."""

from pathlib import Path

import pytest

from proto_transcription.config import TranscriptionConfig
from proto_transcription.exceptions import InvalidAudioFile, InvalidModelName


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig dataclass."""

    def test_valid_config_with_defaults(self, tmp_path: Path) -> None:
        """Test creating config with valid inputs and default model size."""
        # given
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # when
        config = TranscriptionConfig(
            audio_file=str(audio_file),
            output_dir=str(tmp_path),
        )

        # then
        assert config.audio_file == audio_file
        assert config.output_dir == tmp_path
        assert config.model_size == "base"
        assert config.models == ("faster-whisper", "whisperx", "transformers")
        assert config.verbose is False

    def test_valid_config_with_custom_models(self, tmp_path: Path) -> None:
        """Test creating config with custom model selection."""
        # given
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        # when
        config = TranscriptionConfig(
            audio_file=str(audio_file),
            output_dir=str(tmp_path),
            models=("faster-whisper", "whisperx"),
        )

        # then
        assert config.models == ("faster-whisper", "whisperx")

    def test_valid_config_with_custom_model_size(self, tmp_path: Path) -> None:
        """Test creating config with custom model size."""
        # given
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # when
        config = TranscriptionConfig(
            audio_file=str(audio_file),
            output_dir=str(tmp_path),
            model_size="small",
        )

        # then
        assert config.model_size == "small"

    def test_invalid_audio_file_not_found(self, tmp_path: Path) -> None:
        """Test that missing audio file raises InvalidAudioFile."""
        # given
        audio_file = tmp_path / "nonexistent.mp3"

        # when/then
        with pytest.raises(InvalidAudioFile, match="Audio file not found"):
            TranscriptionConfig(
                audio_file=str(audio_file),
                output_dir=str(tmp_path),
            )

    def test_invalid_model_name(self, tmp_path: Path) -> None:
        """Test that invalid model name raises InvalidModelName."""
        # given
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # when/then
        with pytest.raises(InvalidModelName, match="Invalid model name"):
            TranscriptionConfig(
                audio_file=str(audio_file),
                output_dir=str(tmp_path),
                models=("invalid-model",),
            )

    def test_output_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        # given
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        output_dir = tmp_path / "outputs"
        assert not output_dir.exists()

        # when
        config = TranscriptionConfig(
            audio_file=str(audio_file),
            output_dir=str(output_dir),
        )

        # then
        assert config.output_dir == output_dir
        assert output_dir.exists()

    def test_valid_model_sizes(self, tmp_path: Path) -> None:
        """Test that all valid model sizes are accepted."""
        # given
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()
        valid_sizes = ("tiny", "base", "small", "medium", "large")

        # when/then
        for size in valid_sizes:
            config = TranscriptionConfig(
                audio_file=str(audio_file),
                output_dir=str(tmp_path),
                model_size=size,
            )
            assert config.model_size == size

    def test_invalid_model_size(self, tmp_path: Path) -> None:
        """Test that invalid model size is rejected."""
        # given
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # when/then
        with pytest.raises(ValueError, match="Invalid model size"):
            TranscriptionConfig(
                audio_file=str(audio_file),
                output_dir=str(tmp_path),
                model_size="xlarge",
            )

    def test_audio_file_path_converted_to_pathlib(self, tmp_path: Path) -> None:
        """Test that string audio file path is converted to Path object."""
        # given
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # when
        config = TranscriptionConfig(
            audio_file=str(audio_file),
            output_dir=str(tmp_path),
        )

        # then
        assert isinstance(config.audio_file, Path)
        assert config.audio_file == audio_file
