"""Configuration constants and defaults."""

from pathlib import Path

# Whisper.cpp settings
WHISPER_BINARY = "whisper-cli"
DEFAULT_MODEL = "medium"
SUPPORTED_MODELS = ("medium", "large", "large-v2", "large-v3")

# Model paths (Homebrew default location)
MODEL_DIR = Path.home() / ".cache" / "whisper"

# Audio settings
REQUIRED_SAMPLE_RATE = 16000
REQUIRED_CHANNELS = 1
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".opus",
}

# Output settings
DEFAULT_OUTPUT_FORMAT = "txt"
SUPPORTED_OUTPUT_FORMATS = ("txt", "srt", "vtt", "json")
