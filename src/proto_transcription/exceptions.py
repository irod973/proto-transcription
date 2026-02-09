"""Custom exceptions for proto_transcription."""


class ProtoTranscriptionException(Exception):
    """Base exception for proto_transcription."""

    pass


class InvalidAudioFile(ProtoTranscriptionException):
    """Raised when audio file is invalid or not found."""

    pass


class InvalidModelName(ProtoTranscriptionException):
    """Raised when model name is invalid."""

    pass


class FFmpegNotFound(ProtoTranscriptionException):
    """Raised when ffmpeg is not installed or not found."""

    pass


class AudioConversionError(ProtoTranscriptionException):
    """Raised when audio conversion fails."""

    pass


class TranscriptionError(ProtoTranscriptionException):
    """Raised when transcription fails."""

    pass
