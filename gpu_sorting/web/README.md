# Bitonic Sort Web Interface

This is a web-based interface for demonstrating the bitonic sort algorithm. It provides a user-friendly way to interact with sorting algorithms and visualize their performance.

## Features

- **Multiple Input Methods**: 
  - Manual entry of numbers
  - Random number generation
  - File upload of number lists

- **Performance Comparison**: 
  - JavaScript implementation of bitonic sort
  - JavaScript built-in sort (Array.prototype.sort)
  - Timing comparison between both methods

- **Visualization**:
  - Display of unsorted and sorted data
  - Performance metrics and speedup ratios

- **Export Functionality**:
  - Download sorted results as a text file

## Technical Implementation

This web application uses:
- HTML5, CSS3, and vanilla JavaScript for the front-end
- Node.js and Express for the server
- A custom JavaScript implementation of the bitonic sort algorithm

## Requirements

- Node.js (v14 or later recommended)
- npm (comes with Node.js)

## Installation

1. Navigate to the web directory:
   ```
   cd gpu_sorting/web
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Running the Application

### Option 1: Using the batch file (Windows)
From the project root directory, run:
```
run_web_app.bat
```

### Option 2: Manual startup
```
cd gpu_sorting/web
npm install
npm start
```

### Option 3: Start with browser auto-open
```
cd gpu_sorting/web
npm install
npm run start-browser
```

Then open your browser and navigate to:
```
http://localhost:3000
```

## Development

For development with auto-reload:
```
npm run dev
```

## Usage Instructions

1. **Select Input Method**:
   - Manual entry: Type or paste numbers separated by spaces or commas
   - Random generation: Specify count, minimum, and maximum values
   - File upload: Select a text file with numbers separated by spaces or commas

2. **Submit Data**: Click "Submit" to load your data

3. **Review Unsorted Data**: Check the display to confirm your data was loaded correctly

4. **Sort Data**: Click "Sort Data" to perform both bitonic sort and standard sort

5. **Compare Results**:
   - View the time taken by each sorting method
   - See the speedup ratio between them
   - View the sorted data

6. **Download or Reset**:
   - Download the sorted data if needed
   - Reset to start over with new data

## Note on Performance

This web interface uses a JavaScript implementation of the bitonic sort algorithm, which is primarily for demonstration purposes. While it follows the same algorithm as the CUDA C++ implementation, it does not leverage GPU acceleration due to browser limitations.

The real performance benefits of GPU-accelerated bitonic sort are realized in the C++ CUDA implementation. For large datasets, the JavaScript implementation might show poorer performance than the built-in sort due to language and runtime limitations. 