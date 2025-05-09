// ParallelSort - GPU Accelerated Sorting
// Main JavaScript file

// DOM Elements
const startSortingBtn = document.getElementById('start-sorting');
const algorithmSelect = document.getElementById('algorithm');
const dataSizeSelect = document.getElementById('data-size');
const executionModeSelect = document.getElementById('execution-mode');
const visualizationContainer = document.getElementById('visualization-container');
const aboutDialog = document.getElementById('about-dialog');
const helpDialog = document.getElementById('help-dialog');
const aboutLink = document.getElementById('about-link');
const helpLink = document.getElementById('help-link');
const closeDialogBtns = document.querySelectorAll('.dialog-close');

// Application State
const appState = {
  algorithm: 'bitonic',
  dataSize: 1024,
  executionMode: 'gpu',
  isSorting: false,
  sortedData: null,
  performanceMetrics: {
    gpuTime: null,
    cpuTime: null,
    speedup: null
  }
};

// Initialize the application
function initApp() {
  // Add event listeners
  startSortingBtn.addEventListener('click', handleStartSorting);
  algorithmSelect.addEventListener('change', e => { appState.algorithm = e.target.value; });
  dataSizeSelect.addEventListener('change', e => { appState.dataSize = parseInt(e.target.value); });
  executionModeSelect.addEventListener('change', e => { appState.executionMode = e.target.value; });
  
  // Dialog management
  aboutLink.addEventListener('click', () => toggleDialog(aboutDialog, true));
  helpLink.addEventListener('click', () => toggleDialog(helpDialog, true));
  closeDialogBtns.forEach(btn => {
    btn.addEventListener('click', () => toggleDialog(btn.closest('.dialog-overlay'), false));
  });

  // Click outside dialog to close
  document.addEventListener('click', e => {
    const dialogs = document.querySelectorAll('.dialog-overlay.active');
    dialogs.forEach(dialog => {
      if (e.target === dialog) {
        toggleDialog(dialog, false);
      }
    });
  });

  // Keyboard events for dialog
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      const dialogs = document.querySelectorAll('.dialog-overlay.active');
      dialogs.forEach(dialog => toggleDialog(dialog, false));
    }
  });
}

// Toggle dialog visibility
function toggleDialog(dialog, show) {
  if (show) {
    dialog.classList.add('active');
  } else {
    dialog.classList.remove('active');
  }
}

// Handle start sorting button click
function handleStartSorting() {
  if (appState.isSorting) return;
  
  appState.isSorting = true;
  startSortingBtn.disabled = true;
  startSortingBtn.innerHTML = '<span class="material-icons">autorenew</span> Sorting...';
  
  // Clear previous results
  visualizationContainer.innerHTML = '<div class="placeholder">Processing...</div>';
  
  // Generate random data
  const data = generateRandomData(appState.dataSize);
  
  // Perform sorting based on selected algorithm and execution mode
  performSort(data)
    .then(result => {
      appState.sortedData = result.sortedData;
      appState.performanceMetrics = result.metrics;
      displayResults();
    })
    .catch(error => {
      showNotification('Error', error.message, 'error');
      console.error('Sorting error:', error);
    })
    .finally(() => {
      appState.isSorting = false;
      startSortingBtn.disabled = false;
      startSortingBtn.innerHTML = '<span class="material-icons">sort</span> Start Sorting';
    });
}

// Generate random data array
function generateRandomData(size) {
  const data = new Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = Math.floor(Math.random() * 1000);
  }
  return data;
}

// Perform the sorting operation
async function performSort(data) {
  // In a real implementation, this would call the WebAssembly module
  // that interfaces with the GPU or CPU sorting implementations
  
  // For now, we simulate the sorting with different timing for GPU/CPU
  return new Promise((resolve) => {
    const startTime = performance.now();
    
    // Simulate different processing times for GPU vs CPU
    const processingTime = appState.executionMode === 'gpu' 
      ? appState.dataSize * 0.01  // GPU is faster
      : appState.dataSize * 0.1;  // CPU is slower
    
    setTimeout(() => {
      // Sort the data (in a real implementation, this would be done by the WebAssembly module)
      const sortedData = [...data].sort((a, b) => a - b);
      
      const endTime = performance.now();
      const elapsedTime = endTime - startTime;
      
      // Calculate simulated metrics
      let metrics = {};
      if (appState.executionMode === 'gpu') {
        metrics = {
          gpuTime: elapsedTime,
          cpuTime: elapsedTime * 10, // Simulate CPU being 10x slower
          speedup: 10 // GPU is 10x faster in our simulation
        };
      } else {
        metrics = {
          cpuTime: elapsedTime,
          gpuTime: elapsedTime / 10, // Simulate GPU being 10x faster
          speedup: 0.1 // CPU is 10x slower in our simulation
        };
      }
      
      resolve({
        sortedData,
        metrics
      });
    }, processingTime);
  });
}

// Display the sorting results
function displayResults() {
  // Clear the container
  visualizationContainer.innerHTML = '';
  
  // Create performance info section
  const perfInfo = document.createElement('div');
  perfInfo.className = 'performance-info';
  perfInfo.innerHTML = `
    <h3>Performance Results</h3>
    <p><strong>Algorithm:</strong> ${formatAlgorithmName(appState.algorithm)}</p>
    <p><strong>Data Size:</strong> ${appState.dataSize.toLocaleString()} elements</p>
    <p><strong>Execution Mode:</strong> ${appState.executionMode.toUpperCase()}</p>
    <p><strong>${appState.executionMode.toUpperCase()} Time:</strong> ${appState.performanceMetrics[appState.executionMode + 'Time'].toFixed(2)} ms</p>
    <p><strong>Speedup vs ${appState.executionMode === 'gpu' ? 'CPU' : 'GPU'}:</strong> ${appState.performanceMetrics.speedup.toFixed(2)}x</p>
  `;
  
  // Create success animation
  const successAnim = document.createElement('div');
  successAnim.className = 'success-animation';
  successAnim.innerHTML = '<span class="material-icons">check_circle</span>';
  
  // Create sample data visualization (limited to 20 items)
  const dataPreview = document.createElement('div');
  dataPreview.className = 'data-preview';
  
  // Only show a few items for preview
  const maxDisplay = 20;
  const step = Math.max(1, Math.floor(appState.sortedData.length / maxDisplay));
  
  for (let i = 0; i < appState.sortedData.length; i += step) {
    if (dataPreview.children.length >= maxDisplay) break;
    
    const item = document.createElement('div');
    item.className = 'sorted-item';
    item.textContent = appState.sortedData[i];
    dataPreview.appendChild(item);
    
    // Stagger animation
    setTimeout(() => {
      item.classList.add('animated');
    }, 50 * dataPreview.children.length);
  }
  
  // Append all elements to the container
  visualizationContainer.appendChild(perfInfo);
  visualizationContainer.appendChild(successAnim);
  visualizationContainer.appendChild(dataPreview);
  
  // Show success notification
  showNotification('Success', 'Sorting completed successfully!', 'success');
}

// Format algorithm name for display
function formatAlgorithmName(algorithm) {
  const names = {
    'bitonic': 'Bitonic Sort',
    'merge': 'Merge Sort',
    'quick': 'Quick Sort'
  };
  return names[algorithm] || algorithm;
}

// Show notification
function showNotification(title, message, type = 'info') {
  const notificationsContainer = document.querySelector('.notifications-container') || createNotificationsContainer();
  
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  
  notification.innerHTML = `
    <div class="notification-icon">
      <span class="material-icons">
        ${type === 'success' ? 'check_circle' : 
          type === 'error' ? 'error' : 
          type === 'warning' ? 'warning' : 'info'}
      </span>
    </div>
    <div class="notification-content">
      <div class="notification-title">${title}</div>
      <div class="notification-message">${message}</div>
    </div>
    <button class="notification-close">
      <span class="material-icons">close</span>
    </button>
  `;
  
  const closeBtn = notification.querySelector('.notification-close');
  closeBtn.addEventListener('click', () => {
    notification.classList.add('removing');
    setTimeout(() => {
      notification.remove();
    }, 300);
  });
  
  notificationsContainer.appendChild(notification);
  
  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.classList.add('removing');
      setTimeout(() => {
        if (notification.parentNode) {
          notification.remove();
        }
      }, 300);
    }
  }, 5000);
}

// Create notifications container if it doesn't exist
function createNotificationsContainer() {
  const container = document.createElement('div');
  container.className = 'notifications-container';
  document.body.appendChild(container);
  return container;
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);
