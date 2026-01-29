"""Tests for transcriber module."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestDependencyVerification:
    """Tests for dependency verification."""

    @patch("transcribe.transcriber.subprocess.run")
    def test_verify_dependencies_passes_when_all_present(self, mock_run):
        """Should not raise when all dependencies are present."""
        from transcribe.transcriber import Transcriber

        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber(model="medium")
            assert transcriber.model == "medium"

    @patch("transcribe.transcriber.subprocess.run")
    def test_verify_raises_when_whisper_missing(self, mock_run):
        """Should raise DependencyError when whisper-cpp not found."""
        from transcribe.transcriber import Transcriber, DependencyError

        def side_effect(cmd, *args, **kwargs):
            if "whisper-cpp" in cmd:
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        with pytest.raises(DependencyError, match="whisper-cpp"):
            Transcriber()

    @patch("transcribe.transcriber.subprocess.run")
    def test_verify_raises_when_ffmpeg_missing(self, mock_run):
        """Should raise DependencyError when ffmpeg not found."""
        from transcribe.transcriber import Transcriber, DependencyError

        def side_effect(cmd, *args, **kwargs):
            if "ffmpeg" in cmd:
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        with pytest.raises(DependencyError, match="ffmpeg"):
            Transcriber()

    @patch("transcribe.transcriber.subprocess.run")
    def test_verify_raises_when_model_missing(self, mock_run):
        """Should raise DependencyError when model file not found."""
        from transcribe.transcriber import Transcriber, DependencyError

        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(DependencyError, match="[Mm]odel"):
                Transcriber()


class TestTranscribe:
    """Tests for single file transcription."""

    @patch("transcribe.transcriber.subprocess.run")
    @patch("transcribe.transcriber.is_whisper_compatible")
    def test_transcribe_compatible_file(self, mock_compatible, mock_run, tmp_path):
        """Should transcribe compatible file directly."""
        from transcribe.transcriber import Transcriber

        mock_compatible.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber()

        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        result = transcriber.transcribe(audio_file)

        assert result.success is True
        assert result.input_file == audio_file

    @patch("transcribe.transcriber.subprocess.run")
    @patch("transcribe.transcriber.is_whisper_compatible")
    @patch("transcribe.transcriber.convert_to_whisper_format")
    def test_transcribe_converts_incompatible_file(
        self, mock_convert, mock_compatible, mock_run, tmp_path
    ):
        """Should convert incompatible file before transcription."""
        from transcribe.transcriber import Transcriber

        mock_compatible.return_value = False
        mock_convert.return_value = tmp_path / "converted.wav"
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber()

        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        result = transcriber.transcribe(audio_file)

        mock_convert.assert_called_once()
        assert result.success is True

    @patch("transcribe.transcriber.subprocess.run")
    @patch("transcribe.transcriber.is_whisper_compatible")
    def test_transcribe_returns_failure_on_error(
        self, mock_compatible, mock_run, tmp_path
    ):
        """Should return failure result when whisper-cpp fails."""
        from transcribe.transcriber import Transcriber
        import subprocess

        mock_compatible.return_value = True

        # First calls for dependency check, last for transcription
        def run_side_effect(cmd, *args, **kwargs):
            if "whisper-cpp" in cmd and "-f" in cmd:
                raise subprocess.CalledProcessError(1, cmd, stderr="transcription error")
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber()

        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        result = transcriber.transcribe(audio_file)

        assert result.success is False
        assert result.error is not None

    @patch("transcribe.transcriber.subprocess.run")
    @patch("transcribe.transcriber.is_whisper_compatible")
    def test_transcribe_uses_correct_output_path(
        self, mock_compatible, mock_run, tmp_path
    ):
        """Should use specified output path."""
        from transcribe.transcriber import Transcriber

        mock_compatible.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber()

        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        output_file = tmp_path / "output.txt"

        result = transcriber.transcribe(audio_file, output_path=output_file)

        assert result.output_file == output_file


class TestTranscribeBatch:
    """Tests for batch transcription."""

    @patch("transcribe.transcriber.subprocess.run")
    @patch("transcribe.transcriber.is_whisper_compatible")
    def test_batch_processes_all_files(self, mock_compatible, mock_run, tmp_path):
        """Should process all files in batch."""
        from transcribe.transcriber import Transcriber

        mock_compatible.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber()

        files = []
        for i in range(3):
            f = tmp_path / f"test{i}.wav"
            f.touch()
            files.append(f)

        results = transcriber.transcribe_batch(files)

        assert len(results) == 3
        assert all(r.success for r in results)

    @patch("transcribe.transcriber.subprocess.run")
    @patch("transcribe.transcriber.is_whisper_compatible")
    def test_batch_uses_output_directory(self, mock_compatible, mock_run, tmp_path):
        """Should place outputs in specified directory."""
        from transcribe.transcriber import Transcriber

        mock_compatible.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber()

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        audio_file = audio_dir / "test.wav"
        audio_file.touch()

        results = transcriber.transcribe_batch([audio_file], output_dir=output_dir)

        assert results[0].output_file.parent == output_dir

    @patch("transcribe.transcriber.subprocess.run")
    @patch("transcribe.transcriber.is_whisper_compatible")
    def test_batch_continues_on_failure(self, mock_compatible, mock_run, tmp_path):
        """Should continue processing after a file fails."""
        from transcribe.transcriber import Transcriber
        import subprocess

        mock_compatible.return_value = True

        call_count = [0]

        def run_side_effect(cmd, *args, **kwargs):
            if "whisper-cpp" in cmd and "-f" in cmd:
                call_count[0] += 1
                if call_count[0] == 2:  # Fail on second transcription
                    raise subprocess.CalledProcessError(1, cmd, stderr="error")
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with patch.object(Path, "exists", return_value=True):
            transcriber = Transcriber()

        files = []
        for i in range(3):
            f = tmp_path / f"test{i}.wav"
            f.touch()
            files.append(f)

        results = transcriber.transcribe_batch(files)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True
