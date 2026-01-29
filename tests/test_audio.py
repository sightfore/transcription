"""Tests for audio module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDetectFormat:
    """Tests for detect_format function."""

    def test_detect_format_mp3(self):
        """MP3 extension should return 'mp3'."""
        from transcribe.audio import detect_format

        assert detect_format(Path("test.mp3")) == "mp3"

    def test_detect_format_wav(self):
        """WAV extension should return 'wav'."""
        from transcribe.audio import detect_format

        assert detect_format(Path("test.wav")) == "wav"

    def test_detect_format_m4a(self):
        """M4A extension should return 'm4a'."""
        from transcribe.audio import detect_format

        assert detect_format(Path("test.m4a")) == "m4a"

    def test_detect_format_flac(self):
        """FLAC extension should return 'flac'."""
        from transcribe.audio import detect_format

        assert detect_format(Path("test.flac")) == "flac"

    def test_detect_format_case_insensitive(self):
        """Format detection should be case insensitive."""
        from transcribe.audio import detect_format

        assert detect_format(Path("test.MP3")) == "mp3"
        assert detect_format(Path("test.Wav")) == "wav"

    def test_detect_format_unknown_returns_none(self):
        """Unknown extension should return None."""
        from transcribe.audio import detect_format

        assert detect_format(Path("test.xyz")) is None
        assert detect_format(Path("test.txt")) is None


class TestIsWhisperCompatible:
    """Tests for is_whisper_compatible function."""

    @patch("transcribe.audio.subprocess.run")
    def test_compatible_16khz_mono_wav(self, mock_run):
        """16kHz mono WAV should be compatible."""
        from transcribe.audio import is_whisper_compatible

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"codec_name":"pcm_s16le","sample_rate":"16000","channels":1}]}',
        )

        assert is_whisper_compatible(Path("test.wav")) is True

    @patch("transcribe.audio.subprocess.run")
    def test_incompatible_44khz(self, mock_run):
        """44.1kHz audio should not be compatible."""
        from transcribe.audio import is_whisper_compatible

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"codec_name":"pcm_s16le","sample_rate":"44100","channels":1}]}',
        )

        assert is_whisper_compatible(Path("test.wav")) is False

    @patch("transcribe.audio.subprocess.run")
    def test_incompatible_stereo(self, mock_run):
        """Stereo audio should not be compatible."""
        from transcribe.audio import is_whisper_compatible

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"codec_name":"pcm_s16le","sample_rate":"16000","channels":2}]}',
        )

        assert is_whisper_compatible(Path("test.wav")) is False

    @patch("transcribe.audio.subprocess.run")
    def test_incompatible_mp3_format(self, mock_run):
        """MP3 format should not be compatible (not WAV)."""
        from transcribe.audio import is_whisper_compatible

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"codec_name":"mp3","sample_rate":"16000","channels":1}]}',
        )

        assert is_whisper_compatible(Path("test.mp3")) is False

    @patch("transcribe.audio.subprocess.run")
    def test_ffprobe_failure_returns_false(self, mock_run):
        """If ffprobe fails, should return False."""
        from transcribe.audio import is_whisper_compatible

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

        assert is_whisper_compatible(Path("test.wav")) is False


class TestConvertToWhisperFormat:
    """Tests for convert_to_whisper_format function."""

    @patch("transcribe.audio.subprocess.run")
    def test_convert_calls_ffmpeg_with_correct_args(self, mock_run, tmp_path):
        """Conversion should call ffmpeg with correct parameters."""
        from transcribe.audio import convert_to_whisper_format

        mock_run.return_value = MagicMock(returncode=0)
        input_path = tmp_path / "input.mp3"
        input_path.touch()
        output_path = tmp_path / "output.wav"

        convert_to_whisper_format(input_path, output_path)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]

        assert call_args[0] == "ffmpeg"
        assert "-ar" in call_args
        assert "16000" in call_args
        assert "-ac" in call_args
        assert "1" in call_args
        assert str(input_path) in call_args
        assert str(output_path) in call_args

    @patch("transcribe.audio.subprocess.run")
    def test_convert_creates_temp_file_when_no_output(self, mock_run, tmp_path):
        """When no output path given, should create temp file."""
        from transcribe.audio import convert_to_whisper_format

        mock_run.return_value = MagicMock(returncode=0)
        input_path = tmp_path / "input.mp3"
        input_path.touch()

        result = convert_to_whisper_format(input_path)

        assert result.suffix == ".wav"

    @patch("transcribe.audio.subprocess.run")
    def test_convert_raises_on_ffmpeg_failure(self, mock_run, tmp_path):
        """Should raise exception when ffmpeg fails."""
        from transcribe.audio import convert_to_whisper_format, ConversionError

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error")
        input_path = tmp_path / "input.mp3"
        input_path.touch()
        output_path = tmp_path / "output.wav"

        with pytest.raises(ConversionError):
            convert_to_whisper_format(input_path, output_path)
