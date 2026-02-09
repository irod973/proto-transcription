"""Tests for proto_transcription.audio.converter module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_transcription.audio.converter import AudioConverter
from proto_transcription.exceptions import AudioConversionError, FFmpegNotFound


class TestAudioConverter:
    """Tests for AudioConverter class."""

    def test_check_ffmpeg_available(self) -> None:
        """Test that ffmpeg availability check works."""
        # when/then - should not raise
        AudioConverter()
        # If ffmpeg is not installed, this will raise FFmpegNotFound
        # For testing, we mock it to be available

    def test_ffmpeg_not_found(self) -> None:
        """Test that FFmpegNotFound is raised when ffmpeg is missing."""
        # when
        with patch("shutil.which", return_value=None):
            # then
            with pytest.raises(FFmpegNotFound, match="ffmpeg is not installed"):
                AudioConverter()

    def test_is_wav_file(self) -> None:
        """Test is_wav method."""
        # given
        converter = AudioConverter()

        # when/then
        assert converter.is_wav(Path("test.wav")) is True
        assert converter.is_wav(Path("test.WAV")) is True
        assert converter.is_wav(Path("test.mp3")) is False
        assert converter.is_wav(Path("test.m4a")) is False

    def test_convert_mp3_to_wav(self, tmp_path: Path) -> None:
        """Test converting MP3 to WAV."""
        # given
        converter = AudioConverter()
        input_file = tmp_path / "test.mp3"
        input_file.touch()
        output_file = tmp_path / "test.wav"

        # when
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = converter.convert_to_wav(input_file, output_file)

        # then
        assert result == output_file
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args[0]
        assert str(input_file) in call_args
        assert str(output_file) in call_args
        assert "-ar" in call_args  # sample rate flag
        assert "16000" in call_args  # 16kHz

    def test_convert_wav_returns_input_path(self, tmp_path: Path) -> None:
        """Test that converting WAV returns input path unchanged."""
        # given
        converter = AudioConverter()
        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        # when
        result = converter.convert_to_wav(wav_file)

        # then
        assert result == wav_file

    def test_convert_with_ffmpeg_error(self, tmp_path: Path) -> None:
        """Test that AudioConversionError is raised on ffmpeg failure."""
        # given
        converter = AudioConverter()
        input_file = tmp_path / "test.mp3"
        input_file.touch()
        output_file = tmp_path / "test.wav"

        # when
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            # then
            with pytest.raises(AudioConversionError, match="ffmpeg conversion failed"):
                converter.convert_to_wav(input_file, output_file)

    def test_convert_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created if needed."""
        # given
        converter = AudioConverter()
        input_file = tmp_path / "test.mp3"
        input_file.touch()
        output_dir = tmp_path / "subdir" / "output"
        output_file = output_dir / "test.wav"

        # when
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            converter.convert_to_wav(input_file, output_file)

        # then
        assert output_dir.exists()

    def test_get_audio_duration(self, tmp_path: Path) -> None:
        """Test getting audio duration."""
        # given
        converter = AudioConverter()
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # when
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stderr="Duration: 00:05:30.50, start: 0.000000, bitrate: 128 kb/s"
            )
            duration = converter.get_duration(audio_file)

        # then
        assert duration == "00:05:30.50"

    def test_get_audio_duration_error(self, tmp_path: Path) -> None:
        """Test that AudioConversionError is raised if duration query fails."""
        # given
        converter = AudioConverter()
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # when
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # then - no duration in output
            with pytest.raises(AudioConversionError, match="Could not find duration"):
                converter.get_duration(audio_file)
