# proto-transcription

[![check.yml](https://github.com/irod973/proto-transcription/actions/workflows/check.yml/badge.svg)](https://github.com/irod973/proto-transcription/actions/workflows/check.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://irod973.github.io/proto-transcription/)
[![License](https://img.shields.io/github/license/irod973/proto-transcription)](https://github.com/irod973/proto-transcription/blob/main/LICENCE.txt)
[![Release](https://img.shields.io/github/v/release/irod973/proto-transcription)](https://github.com/irod973/proto-transcription/releases)

## Description

Transcribe audio files locally using three Whisper backends — [Faster Whisper](https://github.com/guillaumekln/faster-whisper), [WhisperX](https://github.com/m-bain/whisperx), and [HuggingFace Transformers](https://huggingface.co/docs/transformers/tasks/asr) — and compare their output side-by-side. Designed as a prototyping and evaluation harness for local speech-to-text.

## Installation

**System dependency:** FFmpeg is required for audio conversion.

```bash
brew install ffmpeg
```

**Python dependencies:**

```bash
# Install dev tools
uv run just install

# Install ML inference libraries (torch, faster-whisper, whisperx, transformers)
uv sync --group transcription
```

## Usage

```bash
# Transcribe with all 3 models
proto-transcription transcribe podcast.mp3 --output-dir ./output

# Use specific models only
proto-transcription transcribe podcast.mp3 -m faster-whisper,whisperx

# Choose model size (tiny, base, small, medium, large)
proto-transcription transcribe podcast.mp3 --model-size small

# Verbose output
proto-transcription transcribe podcast.mp3 --verbose
```

Each model writes a separate output file: `output/{audio_stem}_{model}_{size}.txt` with a metadata header and timestamped segments.

## Development

```bash
just check          # run all checks (lint, type, security, coverage)
just check-test     # run pytest only
just format         # format with Ruff
just doc            # generate API docs
```

## Architecture

```
src/proto_transcription/
├── scripts.py              # Click CLI entry point
├── config.py               # Configuration validation
├── exceptions.py           # Exception hierarchy
├── audio/converter.py      # FFmpeg wrapper — converts to 16kHz mono WAV
├── output/writer.py        # Writes timestamped transcription files
└── transcription/
    ├── base.py             # BaseTranscriber abstract interface
    ├── faster_whisper.py   # CTranslate2 backend, CPU + int8
    ├── whisperx.py         # Adds phoneme alignment pass, CPU + int8
    └── transformers.py     # HuggingFace pipeline, MPS on Apple Silicon
```

## Next Steps

### 1. Web UI

The CLI is a solid foundation — the next step is wrapping it in an interactive UI so you can upload audio, kick off transcription, and browse results without touching the terminal.

**FastAPI is the right choice for the backend**, but it's not a UI on its own — it's a Python framework for building HTTP APIs. The typical approach is:

- **FastAPI** handles the backend: accepts audio file uploads, runs transcription (async or as background tasks), and returns results as JSON
- **A frontend** renders the UI: for a quick internal tool, [Streamlit](https://streamlit.io/) or [Gradio](https://gradio.app/) are pure-Python options that need no separate frontend code. For a more polished product, a React/Vue frontend calling the FastAPI endpoints is the standard pattern.

The project already includes a `fastapi` dependency group (`uv sync --group fastapi`) and a `src/fastapi_app/` stub — it's ready to build on.

### 2. Model Evaluation

Running all three models on the same audio is a good start — the next step is measuring them systematically.

#### Metrics

The standard metric for transcription quality is **Word Error Rate (WER)**:

```
WER = (Substitutions + Deletions + Insertions) / Total words in reference
```

Lower is better; 0% is a perfect transcript. A related metric is **Character Error Rate (CER)**, which operates at the character level and is more sensitive to spelling errors and languages without clear word boundaries.

Both require a **reference transcript** (ground truth) to compare against.

#### Evaluation Approach

1. **Collect a test set** — a handful of audio clips with known, manually-verified transcripts covering different speakers, accents, recording conditions, and topics
2. **Run all three models** on each clip and record:
   - **WER / CER** (via a library like [`jiwer`](https://github.com/jitsi/jiwer))
   - **Latency** — wall-clock time for `load_model()` and `transcribe()` separately (model loading is a one-time cost; transcription time scales with audio length, so report it as real-time factor: `transcription_time / audio_duration`)
   - **Memory usage** — peak RAM during inference
3. **Compare outputs qualitatively** — WER misses nuances like punctuation, readability, and how well timestamps align with actual speech

#### Libraries

- [`jiwer`](https://github.com/jitsi/jiwer) — simple WER/CER computation in Python
- [`evaluate`](https://huggingface.co/docs/evaluate) (HuggingFace) — broader suite including WER, supports batch evaluation
- Standard test datasets: [LibriSpeech](https://www.openslr.org/12) (English audiobooks, widely used benchmark), [Common Voice](https://commonvoice.mozilla.org/) (diverse accents/languages)
