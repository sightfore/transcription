"""Audio file detection and conversion utilities."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from transcribe.config import (
    REQUIRED_CHANNELS,
    REQUIRED_SAMPLE_RATE,
    SUPPORTED_AUDIO_EXTENSIONS,
)


class ConversionError(Exception):
    """Raised when audio conversion fails."""

    pass


def detect_format(file_path: Path) -> Optional[str]:
    """
    Detect audio format from file extension.

    Args:
        file_path: Path to the audio file

    Returns:
        Format string (e.g., 'mp3', 'wav') or None if unknown
    """
    ext = file_path.suffix.lower()
    if ext in SUPPORTED_AUDIO_EXTENSIONS:
        return ext[1:]  # Remove leading dot
    return None


def is_whisper_compatible(file_path: Path) -> bool:
    """
    Check if audio file is already in whisper.cpp compatible format.

    Whisper.cpp requires 16kHz, mono, PCM WAV format.

    Args:
        file_path: Path to the audio file

    Returns:
        True if compatible, False otherwise
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(file_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False

    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return False

        audio_stream = streams[0]
        codec = audio_stream.get("codec_name", "")
        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))

        # Must be PCM WAV, 16kHz, mono
        is_pcm = codec in ("pcm_s16le", "pcm_s16be")
        is_16khz = sample_rate == REQUIRED_SAMPLE_RATE
        is_mono = channels == REQUIRED_CHANNELS

        return is_pcm and is_16khz and is_mono

    except (json.JSONDecodeError, KeyError, ValueError):
        return False


def convert_to_whisper_format(
    input_path: Path, output_path: Optional[Path] = None
) -> Path:
    """
    Convert audio to whisper.cpp compatible format (16kHz mono WAV).

    Args:
        input_path: Path to input audio file
        output_path: Path for output file (creates temp file if None)

    Returns:
        Path to converted file

    Raises:
        ConversionError: If conversion fails
    """
    if output_path is None:
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_path = Path(temp_path)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-i",
                str(input_path),
                "-ar",
                str(REQUIRED_SAMPLE_RATE),
                "-ac",
                str(REQUIRED_CHANNELS),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise ConversionError(f"Failed to convert {input_path}: {e.stderr}") from e

    return output_path
