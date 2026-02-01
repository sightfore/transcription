#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHELL_RC=""

# Detect shell config file
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

echo "==> Transcribe CLI Setup"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v whisper-cli &> /dev/null; then
    echo "  [!] whisper-cli not found"
    echo "      Run: brew install whisper-cpp"
    exit 1
else
    echo "  [✓] whisper-cli"
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "  [!] ffmpeg not found"
    echo "      Run: brew install ffmpeg"
    exit 1
else
    echo "  [✓] ffmpeg"
fi

echo ""

# Add alias
ALIAS_LINE="alias transcribe='PYTHONPATH=${SCRIPT_DIR}/src python3 -m transcribe'"

if [[ -n "$SHELL_RC" ]]; then
    if grep -q "alias transcribe=" "$SHELL_RC" 2>/dev/null; then
        echo "Alias already exists in $SHELL_RC"
    else
        echo "" >> "$SHELL_RC"
        echo "# Transcribe CLI" >> "$SHELL_RC"
        echo "$ALIAS_LINE" >> "$SHELL_RC"
        echo "Added alias to $SHELL_RC"
    fi
else
    echo "Could not detect shell config. Add this manually:"
    echo "  $ALIAS_LINE"
fi

echo ""

# Ask about model download
read -p "Download medium model (~1.5GB)? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading model..."
    PYTHONPATH="${SCRIPT_DIR}/src" python3 -m transcribe --bootstrap
fi

echo ""
echo "==> Setup complete!"
echo ""
echo "Run: source $SHELL_RC"
echo "Then: transcribe --help"
