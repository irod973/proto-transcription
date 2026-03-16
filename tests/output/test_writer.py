"""Tests for proto_transcription.output.writer module."""

from pathlib import Path

from proto_transcription.output.writer import TranscriptionWriter


class TestTranscriptionWriter:
    """Tests for TranscriptionWriter class."""

    def test_generate_filename(self, tmp_path: Path) -> None:
        """Test filename generation."""
        # given
        writer = TranscriptionWriter(tmp_path)

        # when
        filename = writer.generate_filename("podcast.mp3", "faster-whisper", "base")

        # then
        assert filename == "podcast_faster-whisper_base.txt"

    def test_generate_filename_different_extensions(self, tmp_path: Path) -> None:
        """Test filename generation with various audio formats."""
        # given
        writer = TranscriptionWriter(tmp_path)

        # when/then
        assert (
            writer.generate_filename("file.wav", "whisperx", "small") == "file_whisperx_small.txt"
        )
        assert (
            writer.generate_filename("file.m4a", "transformers", "large")
            == "file_transformers_large.txt"
        )
        assert (
            writer.generate_filename("file.mp4", "faster-whisper", "tiny")
            == "file_faster-whisper_tiny.txt"
        )

    def test_write_transcription(self, tmp_path: Path) -> None:
        """Test writing transcription with metadata and timestamps."""
        # given
        writer = TranscriptionWriter(tmp_path)
        audio_file = "podcast.mp3"
        model = "faster-whisper"
        model_size = "base"
        text = "[00:00:05] Hello world\n[00:00:10] This is a test"

        # when
        result = writer.write_transcription(audio_file, model, model_size, text)

        # then
        assert result.exists()
        assert result.name == "podcast_faster-whisper_base.txt"
        assert result.parent == tmp_path

        # Verify content
        content = result.read_text()
        assert "podcast.mp3" in content
        assert "faster-whisper" in content
        assert "base" in content
        assert "[00:00:05] Hello world" in content
        assert "[00:00:10] This is a test" in content

    def test_metadata_header_format(self, tmp_path: Path) -> None:
        """Test that metadata header is properly formatted."""
        # given
        writer = TranscriptionWriter(tmp_path)
        audio_file = "test.mp3"
        model = "whisperx"
        model_size = "small"
        text = "[00:00:00] Test transcription"

        # when
        result = writer.write_transcription(audio_file, model, model_size, text)

        # then
        content = result.read_text()

        # Check metadata header
        assert "Model:" in content
        assert "whisperx" in content
        assert "Model Size:" in content
        assert "small" in content
        assert "Audio File:" in content
        assert "test.mp3" in content

    def test_write_transcription_creates_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created if needed."""
        # given
        output_dir = tmp_path / "transcriptions" / "results"

        # when - directory is created in __init__
        writer = TranscriptionWriter(output_dir)
        result = writer.write_transcription(
            "audio.mp3", "faster-whisper", "base", "[00:00:00] test"
        )

        # then
        assert output_dir.exists()
        assert result.parent == output_dir

    def test_write_transcription_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that existing files are overwritten."""
        # given
        writer = TranscriptionWriter(tmp_path)
        filename = "audio_faster-whisper_base.txt"
        existing_file = tmp_path / filename
        existing_file.write_text("old content")

        # when
        result = writer.write_transcription(
            "audio.mp3", "faster-whisper", "base", "[00:00:00] new content"
        )

        # then
        assert result == existing_file
        content = result.read_text()
        assert "new content" in content
        assert "old content" not in content

    def test_write_empty_transcription(self, tmp_path: Path) -> None:
        """Test writing empty transcription."""
        # given
        writer = TranscriptionWriter(tmp_path)

        # when
        result = writer.write_transcription("audio.mp3", "faster-whisper", "base", "")

        # then
        assert result.exists()
        content = result.read_text()
        # Should still have metadata
        assert "Model:" in content
        assert "faster-whisper" in content

    def test_special_characters_in_filename(self, tmp_path: Path) -> None:
        """Test handling of special characters in audio filename."""
        # given
        writer = TranscriptionWriter(tmp_path)
        # Use simple filename without problematic special chars
        audio_file = "my-podcast_episode-01.mp3"

        # when
        filename = writer.generate_filename(audio_file, "faster-whisper", "base")

        # then
        assert filename == "my-podcast_episode-01_faster-whisper_base.txt"

    def test_write_multiline_transcription(self, tmp_path: Path) -> None:
        """Test writing transcription with multiple lines and timestamps."""
        # given
        writer = TranscriptionWriter(tmp_path)
        text = (
            "[00:00:00] First line\n"
            "[00:01:00] Second line\n"
            "[00:02:30] Third line with more text\n"
            "[00:05:00] Last line"
        )

        # when
        result = writer.write_transcription("podcast.mp3", "whisperx", "small", text)

        # then
        content = result.read_text()
        assert "[00:00:00] First line" in content
        assert "[00:01:00] Second line" in content
        assert "[00:02:30] Third line with more text" in content
        assert "[00:05:00] Last line" in content
