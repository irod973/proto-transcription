# Transcriber Implementations Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `load_model()` and `transcribe()` for the three Whisper transcriber stubs (FasterWhisper, WhisperX, Transformers).

**Architecture:** Each transcriber inherits from `BaseTranscriber`, imports its library at module level, loads the model in `load_model()`, and in `transcribe()` calls the library, normalizes the output to `list[dict]` with `start/end/text` keys, then formats with `self.format_segments()`. Library exceptions are wrapped in `TranscriptionError`. All tests mock the library layer to avoid downloading models.

**Tech Stack:** `faster-whisper`, `whisperx`, `transformers` + `torch`, pytest-mock for testing.

---

## Chunk 1: FasterWhisperTranscriber

### Task 1: Implement FasterWhisperTranscriber

**Files:**
- Modify: `src/proto_transcription/transcription/faster_whisper.py`
- Create: `tests/transcription/test_faster_whisper.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/transcription/test_faster_whisper.py`:

```python
"""Tests for FasterWhisperTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_transcription.transcription.faster_whisper import FasterWhisperTranscriber


class TestFasterWhisperTranscriber:
    """Tests for FasterWhisperTranscriber."""

    def test_name(self) -> None:
        """Test transcriber name."""
        transcriber = FasterWhisperTranscriber("base")
        assert transcriber.name == "faster-whisper"

    def test_load_model(self) -> None:
        """Test that load_model instantiates WhisperModel and stores it."""
        transcriber = FasterWhisperTranscriber("base")
        mock_model = MagicMock()

        with patch(
            "proto_transcription.transcription.faster_whisper.WhisperModel",
            return_value=mock_model,
        ) as mock_cls:
            transcriber.load_model()

            mock_cls.assert_called_once_with("base", device="cpu", compute_type="int8")
            assert transcriber.model is mock_model

    def test_transcribe_returns_formatted_text_and_segments(self, tmp_path: Path) -> None:
        """Test transcribe returns formatted text with timestamps and segment dicts."""
        transcriber = FasterWhisperTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        # Mock segments as named-tuple-like objects (faster-whisper returns these)
        seg1 = MagicMock()
        seg1.start = 0.0
        seg1.end = 2.5
        seg1.text = " Hello world"

        seg2 = MagicMock()
        seg2.start = 2.5
        seg2.end = 5.0
        seg2.text = " Goodbye world"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), MagicMock())
        transcriber.model = mock_model

        text, segments = transcriber.transcribe(audio_file)

        assert "[00:00:00]" in text
        assert "Hello world" in text
        assert "[00:00:02]" in text
        assert "Goodbye world" in text

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 2.5, "text": " Hello world"}
        assert segments[1] == {"start": 2.5, "end": 5.0, "text": " Goodbye world"}

    def test_transcribe_calls_model_with_correct_args(self, tmp_path: Path) -> None:
        """Test transcribe calls model.transcribe with str path and beam_size."""
        transcriber = FasterWhisperTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock())
        transcriber.model = mock_model

        transcriber.transcribe(audio_file)

        mock_model.transcribe.assert_called_once_with(str(audio_file), beam_size=5)

    def test_transcribe_wraps_exception_in_transcription_error(self, tmp_path: Path) -> None:
        """Test that library exceptions are wrapped in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = FasterWhisperTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("model failed")
        transcriber.model = mock_model

        with pytest.raises(TranscriptionError, match="model failed"):
            transcriber.transcribe(audio_file)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/transcription/test_faster_whisper.py -v
```

Expected: multiple FAILs — `NotImplementedError` from the stubs, and the error-wrapping test will fail since there's no wrapping yet.

- [ ] **Step 3: Implement FasterWhisperTranscriber**

Replace the full contents of `src/proto_transcription/transcription/faster_whisper.py`:

```python
"""Faster Whisper transcriber implementation using CTranslate2.

Uses the faster-whisper library for faster inference than OpenAI's original
Whisper implementation, thanks to CTranslate2 backend.

Reference: https://github.com/guillaumekln/faster-whisper
"""

from pathlib import Path

from faster_whisper import WhisperModel
from loguru import logger

from proto_transcription.exceptions import TranscriptionError
from proto_transcription.transcription.base import BaseTranscriber


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using Faster Whisper with CTranslate2 backend.

    Provides faster inference than OpenAI's Whisper while maintaining quality.
    Models are automatically downloaded on first use.

    Attributes:
        model_size: Size of model (tiny, base, small, medium, large).
    """

    def __init__(self, model_size: str = "base") -> None:
        """Initialize Faster Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
                Defaults to "base" for good speed/accuracy balance.
        """
        super().__init__(model_size)

    @property
    def name(self) -> str:
        """Get transcriber name.

        Returns:
            "faster-whisper"
        """
        return "faster-whisper"

    def load_model(self) -> None:
        """Load Faster Whisper model.

        Downloads model on first use (~140MB for base model).
        Subsequent runs use cached model. Uses CPU with int8 quantization,
        which is optimal for Apple Silicon (CTranslate2 does not support MPS).

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading Faster Whisper {self.model_size} model")
        try:
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("Faster Whisper model loaded successfully")
        except Exception as e:
            raise TranscriptionError(f"Failed to load Faster Whisper model: {e}") from e

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using Faster Whisper.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys

        Raises:
            TranscriptionError: If transcription fails.
        """
        logger.info(f"Transcribing {audio_file.name} with Faster Whisper")
        try:
            segments_gen, info = self.model.transcribe(str(audio_file), beam_size=5)
            logger.debug(f"Detected language: {info.language} ({info.language_probability:.2f})")

            segments = [
                {"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in segments_gen
            ]
            formatted = self.format_segments(segments)
            logger.info(f"Transcription complete: {len(segments)} segments")
            return formatted, segments
        except Exception as e:
            raise TranscriptionError(f"Faster Whisper transcription failed: {e}") from e
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/transcription/test_faster_whisper.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full quality checks**

```bash
just check
```

Expected: all checks pass (ruff, mypy, bandit, coverage).

- [ ] **Step 6: Commit**

```bash
git add src/proto_transcription/transcription/faster_whisper.py tests/transcription/test_faster_whisper.py
git commit -m "[NO-JIRA] implement FasterWhisperTranscriber"
```

---

## Chunk 2: WhisperXTranscriber

### Task 2: Implement WhisperXTranscriber

**Files:**
- Modify: `src/proto_transcription/transcription/whisperx.py`
- Create: `tests/transcription/test_whisperx.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/transcription/test_whisperx.py`:

```python
"""Tests for WhisperXTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_transcription.transcription.whisperx import WhisperXTranscriber


class TestWhisperXTranscriber:
    """Tests for WhisperXTranscriber."""

    def test_name(self) -> None:
        """Test transcriber name."""
        transcriber = WhisperXTranscriber("base")
        assert transcriber.name == "whisperx"

    def test_load_model(self) -> None:
        """Test that load_model calls whisperx.load_model and stores result."""
        transcriber = WhisperXTranscriber("base")
        mock_model = MagicMock()

        with patch(
            "proto_transcription.transcription.whisperx.whisperx"
        ) as mock_wx:
            mock_wx.load_model.return_value = mock_model
            transcriber.load_model()

            mock_wx.load_model.assert_called_once_with(
                "base", device="cpu", compute_type="int8"
            )
            assert transcriber.model is mock_model

    def test_transcribe_returns_formatted_text_and_segments(self, tmp_path: Path) -> None:
        """Test transcribe returns aligned, formatted segments."""
        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        raw_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.0, "end": 4.5, "text": "How are you"},
        ]
        aligned_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.0, "end": 4.5, "text": "How are you"},
        ]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": raw_segments, "language": "en"}
        transcriber.model = mock_model

        with patch(
            "proto_transcription.transcription.whisperx.whisperx"
        ) as mock_wx:
            mock_align_model = MagicMock()
            mock_metadata = MagicMock()
            mock_wx.load_align_model.return_value = (mock_align_model, mock_metadata)
            mock_wx.align.return_value = {"segments": aligned_segments}
            mock_wx.load_audio.return_value = MagicMock()

            text, segments = transcriber.transcribe(audio_file)

        assert "[00:00:00]" in text
        assert "Hello world" in text
        assert "[00:00:02]" in text
        assert "How are you" in text

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 2.0, "text": "Hello world"}

    def test_transcribe_uses_aligned_result(self, tmp_path: Path) -> None:
        """Test that segments come from the aligned result, not original transcription."""
        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        raw_segments = [{"start": 0.0, "end": 5.0, "text": "Original"}]
        # Alignment refines timestamps
        aligned_segments = [{"start": 0.1, "end": 4.9, "text": "Original"}]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": raw_segments, "language": "en"}
        transcriber.model = mock_model

        with patch(
            "proto_transcription.transcription.whisperx.whisperx"
        ) as mock_wx:
            mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
            mock_wx.align.return_value = {"segments": aligned_segments}
            mock_wx.load_audio.return_value = MagicMock()

            _, segments = transcriber.transcribe(audio_file)

        # Should use aligned timestamps (0.1), not raw (0.0)
        assert segments[0]["start"] == 0.1

    def test_transcribe_wraps_exception_in_transcription_error(self, tmp_path: Path) -> None:
        """Test that library exceptions are wrapped in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = WhisperXTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("model crashed")
        transcriber.model = mock_model

        with patch("proto_transcription.transcription.whisperx.whisperx") as mock_wx:
            mock_wx.load_audio.return_value = MagicMock()
            with pytest.raises(TranscriptionError, match="model crashed"):
                transcriber.transcribe(audio_file)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/transcription/test_whisperx.py -v
```

Expected: multiple FAILs — `NotImplementedError` from the stubs.

- [ ] **Step 3: Implement WhisperXTranscriber**

Replace the full contents of `src/proto_transcription/transcription/whisperx.py`:

```python
"""WhisperX transcriber implementation with alignment.

Uses the whisperx library which adds more accurate word-level timing alignment
compared to base Whisper.

Reference: https://github.com/m-bain/whisperx
"""

from pathlib import Path

import whisperx
from loguru import logger

from proto_transcription.exceptions import TranscriptionError
from proto_transcription.transcription.base import BaseTranscriber


class WhisperXTranscriber(BaseTranscriber):
    """Transcriber using WhisperX with alignment.

    Provides aligned transcriptions with accurate word-level timestamps.
    Models are automatically downloaded on first use. Uses CPU (MPS support
    in WhisperX is limited).

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
        Subsequent runs use cached model. Uses CPU with int8 quantization
        (WhisperX has limited MPS support).

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading WhisperX {self.model_size} model")
        try:
            self.model = whisperx.load_model(
                self.model_size, device="cpu", compute_type="int8"
            )
            logger.info("WhisperX model loaded successfully")
        except Exception as e:
            raise TranscriptionError(f"Failed to load WhisperX model: {e}") from e

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using WhisperX with alignment.

        Runs transcription then applies forced phoneme alignment for more
        accurate word-level timestamps.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys (from aligned result)

        Raises:
            TranscriptionError: If transcription or alignment fails.
        """
        logger.info(f"Transcribing {audio_file.name} with WhisperX")
        try:
            audio = whisperx.load_audio(str(audio_file))
            result = self.model.transcribe(audio, batch_size=16)
            logger.debug(f"Detected language: {result['language']}")

            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device="cpu"
            )
            aligned = whisperx.align(
                result["segments"], model_a, metadata, audio, "cpu",
                return_char_alignments=False,
            )

            segments = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                for seg in aligned["segments"]
            ]
            formatted = self.format_segments(segments)
            logger.info(f"Transcription complete: {len(segments)} segments")
            return formatted, segments
        except Exception as e:
            raise TranscriptionError(f"WhisperX transcription failed: {e}") from e
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/transcription/test_whisperx.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full quality checks**

```bash
just check
```

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/proto_transcription/transcription/whisperx.py tests/transcription/test_whisperx.py
git commit -m "[NO-JIRA] implement WhisperXTranscriber"
```

---

## Chunk 3: TransformersTranscriber

### Task 3: Implement TransformersTranscriber

**Files:**
- Modify: `src/proto_transcription/transcription/transformers.py`
- Create: `tests/transcription/test_transformers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/transcription/test_transformers.py`:

```python
"""Tests for TransformersTranscriber."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_transcription.transcription.transformers import TransformersTranscriber


class TestTransformersTranscriber:
    """Tests for TransformersTranscriber."""

    def test_name(self) -> None:
        """Test transcriber name."""
        transcriber = TransformersTranscriber("base")
        assert transcriber.name == "transformers"

    def test_load_model_uses_mps_when_available(self) -> None:
        """Test load_model uses MPS device when available."""
        transcriber = TransformersTranscriber("base")
        mock_pipe = MagicMock()

        with (
            patch("proto_transcription.transcription.transformers.torch") as mock_torch,
            patch(
                "proto_transcription.transcription.transformers.pipeline",
                return_value=mock_pipe,
            ) as mock_pipeline,
        ):
            mock_torch.backends.mps.is_available.return_value = True

            transcriber.load_model()

            mock_pipeline.assert_called_once_with(
                task="automatic-speech-recognition",
                model="openai/whisper-base",
                device="mps",
            )
            assert transcriber.model is mock_pipe

    def test_load_model_falls_back_to_cpu(self) -> None:
        """Test load_model falls back to CPU when MPS is unavailable."""
        transcriber = TransformersTranscriber("small")
        mock_pipe = MagicMock()

        with (
            patch("proto_transcription.transcription.transformers.torch") as mock_torch,
            patch(
                "proto_transcription.transcription.transformers.pipeline",
                return_value=mock_pipe,
            ) as mock_pipeline,
        ):
            mock_torch.backends.mps.is_available.return_value = False

            transcriber.load_model()

            mock_pipeline.assert_called_once_with(
                task="automatic-speech-recognition",
                model="openai/whisper-small",
                device="cpu",
            )

    def test_transcribe_returns_formatted_text_and_segments(self, tmp_path: Path) -> None:
        """Test transcribe parses HuggingFace chunks into standard segment format."""
        transcriber = TransformersTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        # HuggingFace pipeline returns chunks with (start, end) timestamp tuples
        pipeline_result = {
            "text": "Hello world. How are you.",
            "chunks": [
                {"text": " Hello world.", "timestamp": (0.0, 2.5)},
                {"text": " How are you.", "timestamp": (2.5, 5.0)},
            ],
        }
        mock_pipe = MagicMock(return_value=pipeline_result)
        transcriber.model = mock_pipe

        text, segments = transcriber.transcribe(audio_file)

        assert "[00:00:00]" in text
        assert "Hello world." in text
        assert "[00:00:02]" in text
        assert "How are you." in text

        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 2.5, "text": " Hello world."}
        assert segments[1] == {"start": 2.5, "end": 5.0, "text": " How are you."}

    def test_transcribe_calls_pipeline_with_timestamps(self, tmp_path: Path) -> None:
        """Test transcribe calls pipeline with return_timestamps=True."""
        transcriber = TransformersTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_pipe = MagicMock(return_value={"text": "", "chunks": []})
        transcriber.model = mock_pipe

        transcriber.transcribe(audio_file)

        mock_pipe.assert_called_once_with(str(audio_file), return_timestamps=True)

    def test_transcribe_wraps_exception_in_transcription_error(self, tmp_path: Path) -> None:
        """Test that library exceptions are wrapped in TranscriptionError."""
        from proto_transcription.exceptions import TranscriptionError

        transcriber = TransformersTranscriber("base")
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        mock_pipe = MagicMock(side_effect=RuntimeError("pipeline failed"))
        transcriber.model = mock_pipe

        with pytest.raises(TranscriptionError, match="pipeline failed"):
            transcriber.transcribe(audio_file)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/transcription/test_transformers.py -v
```

Expected: multiple FAILs — `NotImplementedError` from the stubs.

- [ ] **Step 3: Implement TransformersTranscriber**

Replace the full contents of `src/proto_transcription/transcription/transformers.py`:

```python
"""HuggingFace Transformers transcriber implementation.

Uses the HuggingFace Transformers library with the OpenAI Whisper model
from the Hugging Face Model Hub.

Reference: https://huggingface.co/docs/transformers/tasks/asr
"""

from pathlib import Path

import torch
from loguru import logger
from transformers import pipeline

from proto_transcription.exceptions import TranscriptionError
from proto_transcription.transcription.base import BaseTranscriber


class TransformersTranscriber(BaseTranscriber):
    """Transcriber using HuggingFace Transformers and OpenAI Whisper.

    Provides unified interface to Whisper via the popular Transformers library.
    Models are automatically downloaded on first use. Supports MPS acceleration
    on Apple Silicon.

    Attributes:
        model_size: Size of model (tiny, base, small, medium, large).
    """

    def __init__(self, model_size: str = "base") -> None:
        """Initialize Transformers transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large).
                Defaults to "base" for good speed/accuracy balance.
        """
        super().__init__(model_size)

    @property
    def name(self) -> str:
        """Get transcriber name.

        Returns:
            "transformers"
        """
        return "transformers"

    def load_model(self) -> None:
        """Load Transformers Whisper pipeline.

        Downloads model on first use (~140MB for base model).
        Subsequent runs use cached model. Uses MPS on Apple Silicon,
        falls back to CPU otherwise.

        Raises:
            TranscriptionError: If model loading fails.
        """
        logger.info(f"Loading Transformers {self.model_size} model")
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.debug(f"Using device: {device}")
            self.model = pipeline(
                task="automatic-speech-recognition",
                model=f"openai/whisper-{self.model_size}",
                device=device,
            )
            logger.info("Transformers model loaded successfully")
        except Exception as e:
            raise TranscriptionError(f"Failed to load Transformers model: {e}") from e

    def transcribe(self, audio_file: Path) -> tuple[str, list[dict]]:
        """Transcribe audio using Transformers pipeline.

        Args:
            audio_file: Path to WAV/MP3 audio file.

        Returns:
            Tuple of (formatted_text, segments) where:
            - formatted_text: Transcription with [HH:MM:SS] timestamps
            - segments: List of dicts with start, end, text keys

        Raises:
            TranscriptionError: If transcription fails.
        """
        logger.info(f"Transcribing {audio_file.name} with Transformers")
        try:
            result = self.model(str(audio_file), return_timestamps=True)
            segments = [
                {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
                for chunk in result["chunks"]
            ]
            formatted = self.format_segments(segments)
            logger.info(f"Transcription complete: {len(segments)} segments")
            return formatted, segments
        except Exception as e:
            raise TranscriptionError(f"Transformers transcription failed: {e}") from e
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/transcription/test_transformers.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full quality checks**

```bash
just check
```

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/proto_transcription/transcription/transformers.py tests/transcription/test_transformers.py
git commit -m "[NO-JIRA] implement TransformersTranscriber"
```

---

## Chunk 4: Final Verification

### Task 4: Full test suite and smoke test

**Files:** No new files.

- [ ] **Step 1: Run entire test suite**

```bash
just check-test
```

Expected: all 58+ tests pass (43 existing + 14 new).

- [ ] **Step 2: Run full quality checks**

```bash
just check
```

Expected: ruff, mypy, bandit, coverage all pass.

- [ ] **Step 3: Smoke test CLI (optional, requires transcription deps installed)**

If `uv sync --group transcription` has been run and ffmpeg is installed:

```bash
uv run proto-transcription transcribe <path-to-audio.mp3> --models faster-whisper --model-size tiny --verbose
```

Expected: transcription file written to `./transcriptions/`.
