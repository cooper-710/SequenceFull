# Scouting Report Web UI

A simple web interface for generating scouting reports with just a player name.

## Quick Start

1. **Start the web UI:**
   ```bash
   ./start_ui.sh
   ```
   
   Or manually:
   ```bash
   source venv/bin/activate
   pip install Flask
   python3 app.py
   ```

2. **Open your browser:**
   Navigate to `http://127.0.0.1:5000`

3. **Generate a report:**
   - Enter a player name (e.g., "Pete Alonso", "Harrison Bader")
   - Optionally customize team, season start date, or use next series
   - Click "Generate Report"
   - Wait for the report to be generated
   - Download the PDF when ready

## Features

- **Simple Interface**: Just type in a player name and generate a report
- **Real-time Status**: See progress updates as the report is generated
- **Automatic Team Detection**: Uses "AUTO" to automatically detect the player's team
- **Customizable Options**: 
  - Team abbreviation (or AUTO)
  - Season start date
  - Use next series vs next game

## How It Works

The web UI wraps your existing `generate_report.py` script and provides:
- A clean web interface for inputting player names
- Background job processing
- Real-time status updates
- PDF download functionality

Reports are saved to `build/pdf/` directory, just like the command-line version.

## Requirements

- Flask (installed automatically by `start_ui.sh`)
- All existing dependencies for the scouting report engine

