"""Tests for CLI module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_accepts_single_input(self):
        """Parser should accept a single input file."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3"])

        assert args.input == [Path("audio.mp3")]

    def test_parser_accepts_multiple_inputs(self):
        """Parser should accept multiple input files."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["a.mp3", "b.wav", "c.m4a"])

        assert len(args.input) == 3

    def test_parser_accepts_model_option(self):
        """Parser should accept -m/--model option."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["-m", "large", "audio.mp3"])

        assert args.model == "large"

    def test_parser_default_model_is_medium(self):
        """Default model should be medium."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3"])

        assert args.model == "medium"

    def test_parser_accepts_output_option(self):
        """Parser should accept -o/--output option."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3", "-o", "output.txt"])

        assert args.output == "output.txt"

    def test_parser_preserves_trailing_slash_in_output(self):
        """Parser should preserve trailing slash for directory intent."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3", "-o", "out/"])

        assert args.output == "out/"
        assert args.output.endswith("/")

    def test_parser_accepts_format_option(self):
        """Parser should accept -f/--format option."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3", "-f", "srt"])

        assert args.format == "srt"

    def test_parser_default_format_is_txt(self):
        """Default output format should be txt."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3"])

        assert args.format == "txt"

    def test_parser_accepts_language_option(self):
        """Parser should accept -l/--language option."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3", "-l", "es"])

        assert args.language == "es"

    def test_parser_accepts_model_path_option(self):
        """Parser should accept --model-path option."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3", "--model-path", "/path/to/model.bin"])

        assert args.model_path == Path("/path/to/model.bin")

    def test_parser_accepts_verbose_flag(self):
        """Parser should accept -v/--verbose flag."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3", "-v"])

        assert args.verbose == 1

    def test_parser_accepts_double_verbose_flag(self):
        """Parser should accept -vv for debug output."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3", "-vv"])

        assert args.verbose == 2

    def test_parser_default_verbose_is_zero(self):
        """Default verbose level should be 0."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["audio.mp3"])

        assert args.verbose == 0


class TestResolveInputFiles:
    """Tests for input file resolution."""

    def test_resolve_single_file(self, tmp_path):
        """Should return single file in list."""
        from transcribe.cli import resolve_input_files

        audio = tmp_path / "test.mp3"
        audio.touch()

        result = resolve_input_files([audio])

        assert result == [audio]

    def test_resolve_multiple_files(self, tmp_path):
        """Should return all files in list."""
        from transcribe.cli import resolve_input_files

        files = []
        for i in range(3):
            f = tmp_path / f"test{i}.mp3"
            f.touch()
            files.append(f)

        result = resolve_input_files(files)

        assert len(result) == 3

    def test_resolve_directory_expands_audio_files(self, tmp_path):
        """Should expand directory to audio files within."""
        from transcribe.cli import resolve_input_files

        (tmp_path / "a.mp3").touch()
        (tmp_path / "b.wav").touch()
        (tmp_path / "c.txt").touch()  # Not an audio file

        result = resolve_input_files([tmp_path])

        assert len(result) == 2  # Only mp3 and wav

    def test_resolve_skips_nonexistent_files(self, tmp_path, capsys):
        """Should skip files that don't exist with warning."""
        from transcribe.cli import resolve_input_files

        existing = tmp_path / "exists.mp3"
        existing.touch()
        nonexistent = tmp_path / "missing.mp3"

        result = resolve_input_files([existing, nonexistent])

        assert result == [existing]
        captured = capsys.readouterr()
        assert "missing.mp3" in captured.err

    def test_resolve_removes_duplicates(self, tmp_path):
        """Should remove duplicate files."""
        from transcribe.cli import resolve_input_files

        audio = tmp_path / "test.mp3"
        audio.touch()

        result = resolve_input_files([audio, audio])

        assert len(result) == 1


class TestMain:
    """Tests for main entry point."""

    @patch("transcribe.cli.Transcriber")
    def test_main_returns_0_on_success(self, mock_transcriber_class, tmp_path):
        """Should return 0 when all transcriptions succeed."""
        from transcribe.cli import main
        from transcribe.transcriber import TranscriptionResult

        audio = tmp_path / "test.mp3"
        audio.touch()

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            input_file=audio,
            output_file=audio.with_suffix(".txt"),
            model="medium",
            success=True,
        )
        mock_transcriber_class.return_value = mock_transcriber

        result = main([str(audio)])

        assert result == 0

    @patch("transcribe.cli.Transcriber")
    def test_main_returns_1_on_failure(self, mock_transcriber_class, tmp_path):
        """Should return 1 when transcription fails."""
        from transcribe.cli import main
        from transcribe.transcriber import TranscriptionResult

        audio = tmp_path / "test.mp3"
        audio.touch()

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            input_file=audio,
            output_file=audio.with_suffix(".txt"),
            model="medium",
            success=False,
            error="test error",
        )
        mock_transcriber_class.return_value = mock_transcriber

        result = main([str(audio)])

        assert result == 1

    def test_main_returns_1_when_no_files(self, tmp_path, capsys):
        """Should return 1 when no valid input files found."""
        from transcribe.cli import main

        result = main([str(tmp_path / "nonexistent.mp3")])

        assert result == 1
        captured = capsys.readouterr()
        assert "No valid input files" in captured.err

    @patch("transcribe.cli.Transcriber")
    def test_main_passes_model_to_transcriber(self, mock_transcriber_class, tmp_path):
        """Should pass model option to Transcriber."""
        from transcribe.cli import main
        from transcribe.transcriber import TranscriptionResult

        audio = tmp_path / "test.mp3"
        audio.touch()

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            input_file=audio,
            output_file=audio.with_suffix(".txt"),
            model="large",
            success=True,
        )
        mock_transcriber_class.return_value = mock_transcriber

        main(["-m", "large", str(audio)])

        mock_transcriber_class.assert_called_once()
        call_kwargs = mock_transcriber_class.call_args[1]
        assert call_kwargs["model"] == "large"

    @patch("transcribe.cli.Transcriber")
    def test_main_creates_output_directory(self, mock_transcriber_class, tmp_path):
        """Should create output directory if it doesn't exist."""
        from transcribe.cli import main
        from transcribe.transcriber import TranscriptionResult

        audio = tmp_path / "test.mp3"
        audio.touch()
        output_dir = tmp_path / "new_output"

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_batch.return_value = [
            TranscriptionResult(
                input_file=audio,
                output_file=output_dir / "test.txt",
                model="medium",
                success=True,
            )
        ]
        mock_transcriber_class.return_value = mock_transcriber

        # Multiple files triggers batch mode
        audio2 = tmp_path / "test2.mp3"
        audio2.touch()

        main([str(audio), str(audio2), "-o", str(output_dir)])

        assert output_dir.exists()


class TestBootstrap:
    """Tests for --bootstrap functionality."""

    def test_parser_accepts_bootstrap_flag(self):
        """Parser should accept --bootstrap flag without input files."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--bootstrap"])

        assert args.bootstrap is True

    def test_parser_bootstrap_with_model(self):
        """Parser should accept --bootstrap with specific model."""
        from transcribe.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--bootstrap", "-m", "large"])

        assert args.bootstrap is True
        assert args.model == "large"

    @patch("transcribe.cli.download_model")
    def test_bootstrap_downloads_default_model(self, mock_download):
        """Bootstrap should download medium model by default."""
        from transcribe.cli import main

        mock_download.return_value = True

        result = main(["--bootstrap"])

        mock_download.assert_called_once_with("medium")
        assert result == 0

    @patch("transcribe.cli.download_model")
    def test_bootstrap_downloads_specified_model(self, mock_download):
        """Bootstrap should download specified model."""
        from transcribe.cli import main

        mock_download.return_value = True

        result = main(["--bootstrap", "-m", "large"])

        mock_download.assert_called_once_with("large")
        assert result == 0

    @patch("transcribe.cli.download_model")
    def test_bootstrap_returns_1_on_failure(self, mock_download):
        """Bootstrap should return 1 if download fails."""
        from transcribe.cli import main

        mock_download.return_value = False

        result = main(["--bootstrap"])

        assert result == 1


class TestDownloadModel:
    """Tests for model download functionality."""

    @patch("transcribe.cli.subprocess.run")
    def test_download_creates_model_directory(self, mock_run, tmp_path):
        """Should create model directory if it doesn't exist."""
        from transcribe.cli import download_model
        from transcribe import config

        # Temporarily override MODEL_DIR
        original_dir = config.MODEL_DIR
        config.MODEL_DIR = tmp_path / "whisper"

        mock_run.return_value = MagicMock(returncode=0)

        try:
            download_model("medium")
            assert config.MODEL_DIR.exists()
        finally:
            config.MODEL_DIR = original_dir

    @patch("transcribe.cli.subprocess.run")
    def test_download_calls_curl_with_correct_url(self, mock_run, tmp_path):
        """Should call curl with correct Hugging Face URL."""
        from transcribe.cli import download_model
        from transcribe import config

        original_dir = config.MODEL_DIR
        config.MODEL_DIR = tmp_path / "whisper"

        mock_run.return_value = MagicMock(returncode=0)

        try:
            download_model("medium")

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "curl" in call_args
            assert "huggingface.co" in " ".join(call_args)
            assert "ggml-medium.bin" in " ".join(call_args)
        finally:
            config.MODEL_DIR = original_dir

    @patch("transcribe.cli.subprocess.run")
    def test_download_returns_false_on_curl_failure(self, mock_run, tmp_path):
        """Should return False if curl fails."""
        from transcribe.cli import download_model
        from transcribe import config
        import subprocess

        original_dir = config.MODEL_DIR
        config.MODEL_DIR = tmp_path / "whisper"

        mock_run.side_effect = subprocess.CalledProcessError(1, "curl")

        try:
            result = download_model("medium")
            assert result is False
        finally:
            config.MODEL_DIR = original_dir

    @patch("transcribe.cli.subprocess.run")
    def test_download_skips_if_model_exists(self, mock_run, tmp_path):
        """Should skip download if model already exists."""
        from transcribe.cli import download_model
        from transcribe import config

        original_dir = config.MODEL_DIR
        config.MODEL_DIR = tmp_path / "whisper"
        config.MODEL_DIR.mkdir(parents=True)

        # Create existing model file
        model_file = config.MODEL_DIR / "ggml-medium.bin"
        model_file.write_text("fake model")

        try:
            result = download_model("medium")
            mock_run.assert_not_called()
            assert result is True
        finally:
            config.MODEL_DIR = original_dir
