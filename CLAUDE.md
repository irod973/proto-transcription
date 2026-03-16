# proto-transcription

CLI tool for transcribing audio files using three Whisper backends side-by-side.

## Commands

```bash
just                          # list all tasks
just check                    # run all checks (lint, type, security, coverage)
just check-test               # run pytest only
just check-coverage           # run pytest + coverage (default threshold: 50%)
just format                   # format with Ruff
just install                  # install deps + pre-commit hooks

uv sync --group transcription # install ML deps (torch, faster-whisper, whisperx, transformers)
```

## CLI

```bash
proto-transcription transcribe audio.mp3 --output-dir ./output --model-size small --verbose
proto-transcription transcribe audio.mp3 -m faster-whisper,whisperx  # specific models only
```

## Architecture

```
src/proto_transcription/
├── scripts.py          # Click CLI entry point
├── config.py           # TranscriptionConfig dataclass (validates input, creates output dir)
├── exceptions.py       # Exception hierarchy rooted at ProtoTranscriptionException
├── audio/converter.py  # AudioConverter — FFmpeg wrapper, converts to 16kHz mono WAV
├── output/writer.py    # TranscriptionWriter — writes {stem}_{model}_{size}.txt with metadata
└── transcription/
    ├── base.py             # BaseTranscriber: abstract name/load_model()/transcribe()
    ├── faster_whisper.py   # CPU + int8, CTranslate2 backend (no MPS)
    ├── whisperx.py         # CPU + int8, adds phoneme alignment pass
    └── transformers.py     # HuggingFace pipeline, uses MPS on Apple Silicon
```

## Key Patterns

- **BaseTranscriber** normalizes all library output to `list[{"start": float, "end": float, "text": str}]`; use `format_segments()` for `[HH:MM:SS]` output
- **Exception wrapping**: all library exceptions → `TranscriptionError(...) from e`; model-not-loaded guard in all `transcribe()` methods
- **Device selection**: TransformersTranscriber uses MPS if available; Faster Whisper and WhisperX use CPU-only (CTranslate2 doesn't support MPS)
- **Subprocess security**: all `subprocess.run()` calls use `# nosec` (reviewed; no shell=True)

## Testing

Tests mock all heavy ML dependencies at import time (see `conftest.py`); no GPU/model downloads needed.

```bash
uv run pytest tests/transcription/test_faster_whisper.py -v  # single file
```

## Gotchas

- **System dep required**: `brew install ffmpeg` — `AudioConverter` raises `FFmpegNotFound` if missing
- **torchaudio compatibility**: pin `torchaudio<2.9` — `pyannote-audio` (whisperx dep) references `torchaudio.AudioMetaData` removed in 2.9+
- **Logging**: suppressed by default; enabled with `--verbose`. Use `loguru`, never `print`