"""Command-line interface for transcription utility."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from transcribe.config import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    MODEL_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_MODELS,
    SUPPORTED_OUTPUT_FORMATS,
)
from transcribe.transcriber import DependencyError, Transcriber

# Hugging Face model URLs
MODEL_URLS = {
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    "large": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin",
    "large-v2": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
    "large-v3": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
}


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio files using whisper.cpp",
        epilog="Examples:\n"
        "  transcribe audio.mp3\n"
        "  transcribe -m large recording.m4a -o transcript.txt\n"
        "  transcribe --model medium ./recordings/*\n"
        "  transcribe --bootstrap              # Download medium model\n"
        "  transcribe --bootstrap -m large     # Download large model\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments (optional when bootstrapping)
    parser.add_argument(
        "input",
        nargs="*",
        type=Path,
        help="Audio file(s) to transcribe. Supports glob patterns.",
    )

    # Bootstrap mode
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Download the whisper model (use with -m to specify model)",
    )

    # Optional arguments
    parser.add_argument(
        "-m",
        "--model",
        choices=list(SUPPORTED_MODELS),
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file or directory (use trailing / for directory)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=list(SUPPORTED_OUTPUT_FORMATS),
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format (default: {DEFAULT_OUTPUT_FORMAT})",
    )

    parser.add_argument(
        "-l",
        "--language",
        type=str,
        help="Language code (e.g., 'en', 'es'). Auto-detected if not specified.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        help="Custom path to model file",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose output (-v for info, -vv for debug)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


def download_model(model: str) -> bool:
    """
    Download a whisper model from Hugging Face.

    Args:
        model: Model name (medium, large, large-v2, large-v3)

    Returns:
        True if successful, False otherwise
    """
    from transcribe import config

    model_file = config.MODEL_DIR / f"ggml-{model}.bin"

    # Skip if already exists
    if model_file.exists():
        print(f"Model already exists: {model_file}")
        return True

    # Create directory
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    url = MODEL_URLS.get(model)
    if not url:
        print(f"Error: Unknown model '{model}'", file=sys.stderr)
        return False

    print(f"Downloading {model} model...")
    print(f"  URL: {url}")
    print(f"  Destination: {model_file}")

    try:
        subprocess.run(
            [
                "curl",
                "-L",  # Follow redirects
                "-o",
                str(model_file),
                "--progress-bar",
                url,
            ],
            check=True,
        )
        print(f"Download complete: {model_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        # Clean up partial download
        if model_file.exists():
            model_file.unlink()
        return False


def resolve_input_files(input_args: list[Path]) -> list[Path]:
    """
    Resolve input arguments to list of existing audio files.

    Handles directories and validates files exist.
    """
    files = []

    for path in input_args:
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                files.append(path)
        elif path.is_dir():
            # Process all audio files in directory
            for ext in SUPPORTED_AUDIO_EXTENSIONS:
                files.extend(path.glob(f"*{ext}"))
        else:
            print(f"Warning: {path} not found, skipping", file=sys.stderr)

    # Remove duplicates and sort
    return sorted(set(files))


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    verbose = args.verbose

    # Handle bootstrap mode
    if args.bootstrap:
        success = download_model(args.model)
        return 0 if success else 1

    # Normal transcription mode - require input files
    if not args.input:
        parser.error("the following arguments are required: input")

    # Resolve input files
    input_files = resolve_input_files(args.input)
    if not input_files:
        print("Error: No valid input files found", file=sys.stderr)
        return 1

    # Show config info
    print(f"Model: {args.model}")
    print(f"Input: {len(input_files)} file(s)")
    if verbose >= 1:
        for f in input_files:
            print(f"  - {f}")
    print()

    # Determine output handling
    output_dir = None
    output_file = None
    if args.output:
        output_path = Path(args.output)
        # Trailing slash = directory intent, or existing dir, or multiple files
        is_dir_intent = args.output.endswith("/") or args.output.endswith("\\")
        if is_dir_intent or output_path.is_dir() or len(input_files) > 1:
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_file = output_path

    # Initialize transcriber
    try:
        transcriber = Transcriber(
            model=args.model,
            model_path=args.model_path,
            language=args.language,
            verbose=verbose,
        )
    except DependencyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if verbose >= 1:
        print(f"Model path: {transcriber.model_path}")
        if args.language:
            print(f"Language: {args.language}")
        print()

    # Run transcription
    if len(input_files) == 1 and not output_dir:
        # Single file mode
        result = transcriber.transcribe(input_files[0], output_file, args.format)
        results = [result]
    else:
        # Batch mode
        results = transcriber.transcribe_batch(input_files, output_dir, args.format)

    # Summary
    print()
    success_count = sum(1 for r in results if r.success)
    print(f"Completed: {success_count}/{len(results)} files transcribed")

    # Show output locations
    if verbose >= 1:
        print("\nOutput files:")
        for r in results:
            status = "✓" if r.success else "✗"
            print(f"  {status} {r.output_file}")

    return 0 if success_count == len(results) else 1
