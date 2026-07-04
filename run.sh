#!/bin/bash
# Run the PDF chatbot using the local virtual environment
# Usage: bash run.sh /path/to/your/file.pdf

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: bash run.sh /path/to/your/file.pdf"
    exit 1
fi

cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/venv/bin/activate"
python "$SCRIPT_DIR/app.py" "$1"
