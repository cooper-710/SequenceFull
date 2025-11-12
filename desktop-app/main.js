const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;

// Get the root directory - in development it's the parent, in production check extraResources
function getRootDir() {
  // In packaged app, resources are in extraResources
  if (process.resourcesPath && fs.existsSync(path.join(process.resourcesPath, 'src'))) {
    return process.resourcesPath;
  }
  // In development, use parent directory
  return path.resolve(__dirname, '..');
}

const ROOT_DIR = getRootDir();

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    backgroundColor: '#f5f5f5',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show: false
  });

  mainWindow.loadFile('index.html');

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers
ipcMain.handle('generate-report', async (event, options) => {
  const { hitterName, team = 'AUTO', seasonStart = '2025-03-20', useNextSeries = false } = options;

  return new Promise((resolve, reject) => {
    const venvPython = path.join(ROOT_DIR, 'venv', 'bin', 'python3');
    const scriptPath = path.join(ROOT_DIR, 'src', 'generate_report.py');
    const templatePath = path.join(ROOT_DIR, 'src', 'templates', 'hitter_report.html');
    const outDir = path.join(ROOT_DIR, 'build', 'pdf');

    // Ensure output directory exists
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }

    // Use venv Python if available, otherwise use system Python
    const pythonPath = fs.existsSync(venvPython) ? venvPython : 'python3';

    const args = [
      scriptPath,
      '--team', team,
      '--hitter', hitterName,
      '--season_start', seasonStart,
      '--out', outDir,
      '--template', templatePath
    ];

    if (useNextSeries) {
      args.push('--use-next-series');
    }

    const process = spawn(pythonPath, args, {
      cwd: path.join(ROOT_DIR, 'src'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => {
      stdout += data.toString();
      // Send progress updates to renderer
      event.sender.send('report-progress', { type: 'stdout', data: data.toString() });
    });

    process.stderr.on('data', (data) => {
      stderr += data.toString();
      event.sender.send('report-progress', { type: 'stderr', data: data.toString() });
    });

    process.on('close', (code) => {
      if (code === 0) {
        // Parse output to find PDF path
        let pdfPath = null;
        const lines = stdout.split('\n');
        for (const line of lines) {
          if (line.includes('Saved report:')) {
            pdfPath = line.split('Saved report:')[1].trim();
            break;
          }
        }

        // If not found in output, try to find by name
        if (!pdfPath || !fs.existsSync(pdfPath)) {
          const safeName = hitterName.replace(/[^a-zA-Z0-9]/g, '_');
          const pdfFiles = fs.readdirSync(outDir)
            .filter(f => f.endsWith('.pdf') && f.includes(safeName))
            .map(f => path.join(outDir, f))
            .filter(f => fs.existsSync(f))
            .sort((a, b) => fs.statSync(b).mtime - fs.statSync(a).mtime);

          if (pdfFiles.length > 0) {
            pdfPath = pdfFiles[0];
          }
        }

        if (pdfPath && fs.existsSync(pdfPath)) {
          resolve({
            success: true,
            pdfPath: pdfPath,
            filename: path.basename(pdfPath)
          });
        } else {
          reject(new Error('Report generated but PDF file not found'));
        }
      } else {
        reject(new Error(stderr || stdout || `Process exited with code ${code}`));
      }
    });

    process.on('error', (error) => {
      reject(new Error(`Failed to start process: ${error.message}`));
    });
  });
});

ipcMain.handle('open-pdf', async (event, pdfPath) => {
  try {
    await shell.openPath(pdfPath);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('show-save-dialog', async (event, defaultPath) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultPath,
    filters: [
      { name: 'PDF Files', extensions: ['pdf'] }
    ]
  });

  if (!result.canceled) {
    return { success: true, path: result.filePath };
  }
  return { success: false };
});

ipcMain.handle('open-folder', async (event, folderPath) => {
  try {
    await shell.openPath(folderPath);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-app-version', () => {
  return app.getVersion();
});

