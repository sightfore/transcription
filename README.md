# transcribe

A simple CLI for audio transcription using whisper.cpp on macOS.

## Setup

```bash
# Install prerequisites
brew install whisper-cpp ffmpeg

# Run setup (adds alias, optionally downloads model)
./setup.sh

# Reload shell
source ~/.zshrc
```

## Quick Start

```bash
transcribe --bootstrap      # Download model (first time only)
transcribe audio.mp3        # Transcribe a file
```

## Usage

```bash
# Single file
transcribe audio.mp3

# Specify output
transcribe audio.mp3 -o transcript.txt

# Batch processing
transcribe ./recordings/*
transcribe ./recordings/* -o ./transcripts/

# Use large model
transcribe -m large interview.m4a

# Output formats: txt, srt, vtt, json
transcribe lecture.wav -f srt

# Specify language (skip auto-detection)
transcribe spanish.mp3 -l es
```

## Bootstrap

Download whisper models from Hugging Face:

```bash
# Medium model (~1.5GB) - default
transcribe --bootstrap

# Large model (~3GB)
transcribe --bootstrap -m large

# Large v3
transcribe --bootstrap -m large-v3
```

Models are stored in `~/.cache/whisper/`.

## Options

| Option | Description |
|--------|-------------|
| `-m, --model` | Model: medium, large, large-v2, large-v3 (default: medium) |
| `-o, --output` | Output file or directory |
| `-f, --format` | Output format: txt, srt, vtt, json (default: txt) |
| `-l, --language` | Language code (e.g., en, es) |
| `--model-path` | Custom model file path |
| `--bootstrap` | Download the whisper model |
| `-v, --verbose` | Verbose output |

## Supported Audio Formats

wav, mp3, m4a, flac, ogg, aac, wma, opus

Audio is automatically converted to 16kHz mono WAV for whisper.cpp.

## License

Apache 2.0
