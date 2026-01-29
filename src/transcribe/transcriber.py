"""Core transcription functionality using whisper.cpp."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transcribe.audio import convert_to_whisper_format, is_whisper_compatible
from transcribe.config import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    MODEL_DIR,
    WHISPER_BINARY,
)


class DependencyError(Exception):
    """Raised when a required external dependency is missing."""

    pass


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    input_file: Path
    output_file: Path
    model: str
    success: bool
    error: Optional[str] = None


class Transcriber:
    """Handles transcription using whisper.cpp binary."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        model_path: Optional[Path] = None,
        language: Optional[str] = None,
        verbose: int = 0,
    ):
        self.model = model
        self.model_path = model_path or self._default_model_path(model)
        self.language = language
        self.verbose = verbose
        self._verify_dependencies()

    def _default_model_path(self, model: str) -> Path:
        """Get default model path for given model name."""
        model_file = f"ggml-{model}.bin"
        return MODEL_DIR / model_file

    def _verify_dependencies(self) -> None:
        """Verify whisper-cpp and ffmpeg are available."""
        # Check whisper-cpp
        result = subprocess.run(
            ["which", WHISPER_BINARY], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise DependencyError(
                f"{WHISPER_BINARY} not found. Install with: brew install whisper-cpp"
            )

        # Check ffmpeg
        result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
        if result.returncode != 0:
            raise DependencyError(
                "ffmpeg not found. Install with: brew install ffmpeg"
            )

        # Check model file
        if not self.model_path.exists():
            raise DependencyError(
                f"Model not found at {self.model_path}. "
                f"Download with: whisper-cpp-download-ggml-model {self.model}"
            )

    def transcribe(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to input audio file
            output_path: Path for output (default: same as input with new ext)
            output_format: Output format (txt, srt, vtt, json)

        Returns:
            TranscriptionResult with status and paths
        """
        # Determine output path
        if output_path is None:
            output_path = audio_path.with_suffix(f".{output_format}")

        print(f"Transcribing: {audio_path.name}")

        # Convert if needed
        temp_wav = None
        if not is_whisper_compatible(audio_path):
            if self.verbose >= 1:
                print(f"  Converting to 16kHz WAV...")
            temp_wav = convert_to_whisper_format(audio_path)
            wav_path = temp_wav
            if self.verbose >= 1:
                print(f"  Converted: {temp_wav}")
        else:
            wav_path = audio_path
            if self.verbose >= 1:
                print(f"  Audio already compatible")

        # Build whisper-cpp command
        # Output file prefix (whisper-cpp adds the extension)
        output_prefix = output_path.with_suffix("")

        cmd = [
            WHISPER_BINARY,
            "-m",
            str(self.model_path),
            "-f",
            str(wav_path),
            f"-o{output_format}",
            "-of",
            str(output_prefix),
        ]

        if self.language:
            cmd.extend(["-l", self.language])

        if self.verbose >= 2:
            print(f"  Command: {' '.join(cmd)}")

        # Run transcription
        try:
            if self.verbose >= 1:
                print(f"  Running whisper-cpp...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if self.verbose >= 2 and result.stderr:
                print(f"  Whisper output: {result.stderr[:200]}")
            print(f"  ✓ Done: {output_path.name}")
            return TranscriptionResult(
                input_file=audio_path,
                output_file=output_path,
                model=self.model,
                success=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e.stderr or str(e)}")
            return TranscriptionResult(
                input_file=audio_path,
                output_file=output_path,
                model=self.model,
                success=False,
                error=e.stderr or str(e),
            )
        finally:
            # Clean up temp file
            if temp_wav and temp_wav.exists():
                if self.verbose >= 2:
                    print(f"  Cleaning up: {temp_wav}")
                temp_wav.unlink()

    def transcribe_batch(
        self,
        audio_paths: list[Path],
        output_dir: Optional[Path] = None,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of input audio file paths
            output_dir: Directory for outputs (default: same as input files)
            output_format: Output format for all files

        Returns:
            List of TranscriptionResult objects
        """
        results = []
        total = len(audio_paths)

        for i, audio_path in enumerate(audio_paths, 1):
            if output_dir:
                output_path = output_dir / audio_path.with_suffix(
                    f".{output_format}"
                ).name
            else:
                output_path = None

            print(f"[{i}/{total}] ", end="")
            result = self.transcribe(audio_path, output_path, output_format)
            results.append(result)

        return results
