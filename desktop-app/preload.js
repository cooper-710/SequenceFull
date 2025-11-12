const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  generateReport: (options) => ipcRenderer.invoke('generate-report', options),
  openPDF: (pdfPath) => ipcRenderer.invoke('open-pdf', pdfPath),
  showSaveDialog: (defaultPath) => ipcRenderer.invoke('show-save-dialog', defaultPath),
  openFolder: (folderPath) => ipcRenderer.invoke('open-folder', folderPath),
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  onReportProgress: (callback) => {
    ipcRenderer.on('report-progress', (event, data) => callback(data));
  },
  removeReportProgressListener: () => {
    ipcRenderer.removeAllListeners('report-progress');
  }
});

