# Desktop Application

A native desktop application built with Electron for generating scouting reports.

## Quick Start

### First Time Setup

1. **Install Node.js** (if not already installed)
   ```bash
   # macOS
   brew install node
   
   # Or download from https://nodejs.org/
   ```

2. **Start the app:**
   ```bash
   ./start-app.sh
   ```
   
   Or manually:
   ```bash
   cd desktop-app
   npm install
   npm start
   ```

## Using the App

1. Enter a player name (e.g., "Pete Alonso", "Harrison Bader")
2. Optionally customize:
   - Team (or leave as "AUTO" for auto-detection)
   - Season start date
   - Use next series instead of next game
3. Click "Generate Report"
4. Wait for the report to be generated (progress shown in real-time)
5. Click "Open Report" to view the PDF, or "Open Reports Folder" to see all reports

## Building for Distribution

### macOS
```bash
cd desktop-app
npm run build:mac
```

### Windows
```bash
cd desktop-app
npm run build:win
```

### Linux
```bash
cd desktop-app
npm run build:linux
```

Built applications will be in the `desktop-app/dist` folder.

## Development

To run with DevTools open:
```bash
cd desktop-app
npm run dev
```

## Features

- ✅ Native desktop application (not a web server)
- ✅ Modern, clean UI
- ✅ Real-time progress updates
- ✅ Automatic PDF opening
- ✅ Open reports folder
- ✅ Works on macOS, Windows, and Linux
- ✅ Cross-platform builds

