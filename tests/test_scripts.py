# %% IMPORTS

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from proto_transcription.scripts import cli

# %% FUNCTIONS


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self) -> None:
        """Test CLI help output."""
        # given
        runner = CliRunner()

        # when
        result = runner.invoke(cli, ["--help"])

        # then
        assert result.exit_code == 0
        assert "transcribe" in result.output

    def test_transcribe_help(self) -> None:
        """Test transcribe command help."""
        # given
        runner = CliRunner()

        # when
        result = runner.invoke(cli, ["transcribe", "--help"])

        # then
        assert result.exit_code == 0
        assert "--output-dir" in result.output
        assert "--models" in result.output
        assert "--model-size" in result.output
        assert "--verbose" in result.output

    def test_transcribe_missing_audio_file(self) -> None:
        """Test transcribe with missing audio file."""
        # given
        runner = CliRunner()

        # when
        result = runner.invoke(cli, ["transcribe", "nonexistent.mp3"])

        # then
        assert result.exit_code != 0

    def test_transcribe_with_valid_audio_file(
        self, sample_audio_file: Path, temp_output_dir: Path
    ) -> None:
        """Test transcribe with valid audio file."""
        # given
        runner = CliRunner()

        # when - mock the transcribers
        with patch("proto_transcription.scripts.AudioConverter") as mock_converter:
            with patch("proto_transcription.scripts.TranscriptionWriter") as mock_writer:
                with patch("proto_transcription.scripts.FasterWhisperTranscriber") as mock_fw:
                    with patch("proto_transcription.scripts.WhisperXTranscriber") as mock_wx:
                        with patch(
                            "proto_transcription.scripts.TransformersTranscriber"
                        ) as mock_tf:
                            # Setup mocks
                            mock_converter_instance = MagicMock()
                            mock_converter_instance.convert_to_wav.return_value = sample_audio_file
                            mock_converter.return_value = mock_converter_instance

                            mock_writer_instance = MagicMock()
                            mock_writer.return_value = mock_writer_instance

                            # Mock transcribers
                            for mock_class in [mock_fw, mock_wx, mock_tf]:
                                mock_instance = MagicMock()
                                mock_instance.name = "test"
                                mock_instance.transcribe.return_value = ("[00:00:00] test", [])
                                mock_class.return_value = mock_instance

                            result = runner.invoke(
                                cli,
                                [
                                    "transcribe",
                                    str(sample_audio_file),
                                    "--output-dir",
                                    str(temp_output_dir),
                                ],
                            )

        # then
        assert result.exit_code == 0

    def test_transcribe_with_custom_models(
        self, sample_audio_file: Path, temp_output_dir: Path
    ) -> None:
        """Test transcribe with custom model selection."""
        # given
        runner = CliRunner()

        # when
        with patch("proto_transcription.scripts.AudioConverter"):
            with patch("proto_transcription.scripts.TranscriptionWriter"):
                with patch("proto_transcription.scripts.FasterWhisperTranscriber") as mock_fw:
                    mock_instance = MagicMock()
                    mock_instance.name = "faster-whisper"
                    mock_instance.transcribe.return_value = ("[00:00:00] test", [])
                    mock_fw.return_value = mock_instance

                    result = runner.invoke(
                        cli,
                        [
                            "transcribe",
                            str(sample_audio_file),
                            "--output-dir",
                            str(temp_output_dir),
                            "--models",
                            "faster-whisper",
                        ],
                    )

        # then
        assert result.exit_code == 0

    def test_transcribe_with_model_size(
        self, sample_audio_file: Path, temp_output_dir: Path
    ) -> None:
        """Test transcribe with custom model size."""
        # given
        runner = CliRunner()

        # when
        with patch("proto_transcription.scripts.AudioConverter"):
            with patch("proto_transcription.scripts.TranscriptionWriter"):
                with patch("proto_transcription.scripts.FasterWhisperTranscriber") as mock_fw:
                    mock_instance = MagicMock()
                    mock_instance.model_size = "small"
                    mock_instance.transcribe.return_value = ("[00:00:00] test", [])
                    mock_fw.return_value = mock_instance

                    result = runner.invoke(
                        cli,
                        [
                            "transcribe",
                            str(sample_audio_file),
                            "--output-dir",
                            str(temp_output_dir),
                            "--model-size",
                            "small",
                            "--models",
                            "faster-whisper",
                        ],
                    )

        # then
        assert result.exit_code == 0

    def test_transcribe_verbose_logging(
        self, sample_audio_file: Path, temp_output_dir: Path
    ) -> None:
        """Test that verbose flag enables verbose logging."""
        # given
        runner = CliRunner()

        # when
        with patch("proto_transcription.scripts.AudioConverter"):
            with patch("proto_transcription.scripts.TranscriptionWriter"):
                with patch("proto_transcription.scripts.FasterWhisperTranscriber") as mock_fw:
                    mock_instance = MagicMock()
                    mock_instance.transcribe.return_value = ("[00:00:00] test", [])
                    mock_fw.return_value = mock_instance

                    result = runner.invoke(
                        cli,
                        [
                            "transcribe",
                            str(sample_audio_file),
                            "--output-dir",
                            str(temp_output_dir),
                            "--verbose",
                            "--models",
                            "faster-whisper",
                        ],
                    )

        # then
        assert result.exit_code == 0
