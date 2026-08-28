#!/bin/bash
# Run the PDF chatbot CLI using the local virtual environment
# Usage: ./run.sh /path/to/your/file.pdf

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$1" ]; then
    echo "❌ Error: Please provide a path to a PDF file."
    echo "Usage: ./run.sh /path/to/your/file.pdf"
    exit 1
fi

cd "$SCRIPT_DIR"
if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
    "$SCRIPT_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

source "$SCRIPT_DIR/venv/bin/activate"
python "$SCRIPT_DIR/cli.py" "$1"
