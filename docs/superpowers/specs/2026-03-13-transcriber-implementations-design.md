# Transcriber Implementations Design

**Date:** 2026-03-13
**Branch:** feature/initial-prototyping
**Status:** Approved

## Overview

Implement the three Whisper transcriber stubs (`FasterWhisperTranscriber`, `WhisperXTranscriber`, `TransformersTranscriber`) that were scaffolded in the previous session. Each transcriber implements `load_model()` and `transcribe()` from the `BaseTranscriber` abstract class.

## Constraints

- **Target device:** Apple Silicon Mac (M1/M2/M3/M4)
- **No diarization:** WhisperX alignment only, no speaker diarization (no HF token required)
- **Standard imports:** Libraries are imported normally at module level; `ImportError` propagates naturally if not installed
- **Error handling:** Wrap library exceptions in `TranscriptionError` with the original exception as context (matches the documented `BaseTranscriber` interface). CLI already catches per-model exceptions and continues with remaining models.

## Implementation Details

### 1. FasterWhisperTranscriber

**File:** `src/proto_transcription/transcription/faster_whisper.py`

**load_model():**
- Instantiate `faster_whisper.WhisperModel(self.model_size, device="cpu", compute_type="int8")`
- CTranslate2 backend does not support MPS; CPU with int8 quantization is the best option on Apple Silicon

**transcribe():**
- Call `self.model.transcribe(str(audio_file), beam_size=5)`
- Returns `(segments_generator, info)` where segments are named objects with `.start`, `.end`, `.text`
- Convert generator to list, extract segment dicts `{"start", "end", "text"}`
- Format with `self.format_segments()` and return `(formatted_text, segments)`

### 2. WhisperXTranscriber

**File:** `src/proto_transcription/transcription/whisperx.py`

**load_model():**
- Call `whisperx.load_model(self.model_size, device="cpu", compute_type="int8")`
- WhisperX has limited MPS support; CPU is safer

**transcribe():**
- Load audio: `audio = whisperx.load_audio(str(audio_file))`
- Transcribe: `result = self.model.transcribe(audio, batch_size=16)`
- Align: Load alignment model with `whisperx.load_align_model(language_code=result["language"], device="cpu")`, then call `aligned = whisperx.align(result["segments"], model_a, metadata, audio, "cpu")`. Note: `whisperx.align()` returns a **new** result dict with refined timestamps.
- Extract segments from the **aligned** result (`aligned["segments"]`), not the original `result["segments"]`
- Format with `self.format_segments()` and return `(formatted_text, segments)`
- No diarization step

### 3. TransformersTranscriber

**File:** `src/proto_transcription/transcription/transformers.py`

**load_model():**
- Detect device: `"mps"` if `torch.backends.mps.is_available()` else `"cpu"`
- Create pipeline: `pipeline(task="automatic-speech-recognition", model=f"openai/whisper-{self.model_size}", device=device)`
- Transformers pipeline supports MPS on Apple Silicon

**transcribe():**
- Call `self.model(str(audio_file), return_timestamps=True)`
- Returns dict with `"text"` and `"chunks"` where each chunk has `"text"` and `"timestamp": (start, end)`
- Convert chunks to segment dicts `{"start", "end", "text"}`
- Format with `self.format_segments()` and return `(formatted_text, segments)`

## Testing Strategy

All three transcribers require large model downloads (~140MB+ each), so tests mock the library calls:

- **FasterWhisperTranscriber:** Mock `faster_whisper.WhisperModel` and its `transcribe()` method returning mock segment objects with `.start`, `.end`, `.text` attributes
- **WhisperXTranscriber:** Mock `whisperx.load_model`, `whisperx.load_audio`, `whisperx.load_align_model`, and `whisperx.align`; return dicts matching WhisperX's output format
- **TransformersTranscriber:** Mock `transformers.pipeline`; return dict with `"text"` and `"chunks"` matching HuggingFace format

Each test verifies:
1. `load_model()` sets `self.model` correctly
2. `transcribe()` returns `(formatted_text, segments)` in the expected format
3. Segments contain `start`, `end`, `text` keys
4. Formatted text contains `[HH:MM:SS]` timestamps
