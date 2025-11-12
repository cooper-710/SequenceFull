#!/bin/bash
# Start the Scouting Report Web UI

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install Flask if not already installed
python3 -m pip install Flask --quiet 2>/dev/null || true

# Start the Flask app
echo "Starting Scouting Report Web UI..."
echo "Open http://127.0.0.1:5000 in your browser"
echo ""
python3 app.py

