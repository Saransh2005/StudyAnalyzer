#!/bin/bash
# Run the PDF chatbot using the local virtual environment

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/venv/bin/activate"
python "$SCRIPT_DIR/app.py"
