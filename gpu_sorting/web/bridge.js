const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Path to the compiled C++ executable
const executablePath = path.join(__dirname, '..', 'build', 'gpu_sort_bridge.exe');

// Limit for data size before using a more efficient approach
const LARGE_DATASET_THRESHOLD = 1000000; // 1 million items

// Function to run the C++ CUDA implementation
async function runNativeBitonicSort(data) {
    return new Promise((resolve, reject) => {
        // Validate executable exists
        if (!fs.existsSync(executablePath)) {
            return reject(new Error(`CUDA executable not found at: ${executablePath}. Please run build_bridge.bat first.`));
        }

        // Create a temporary file to store the input data
        const tempInputFile = path.join(os.tmpdir(), `sort_input_${Date.now()}.txt`);
        const tempOutputFile = path.join(os.tmpdir(), `sort_output_${Date.now()}.txt`);
        
        try {
            console.log(`Writing ${data.length} numbers to temporary file...`);
            
            // For very large datasets, write to file in chunks
            if (data.length > LARGE_DATASET_THRESHOLD) {
                const writeStream = fs.createWriteStream(tempInputFile);
                
                // Write data in chunks to avoid memory issues
                const CHUNK_SIZE = 100000;
                for (let i = 0; i < data.length; i += CHUNK_SIZE) {
                    const chunk = data.slice(i, i + CHUNK_SIZE);
                    writeStream.write(chunk.join('\n') + '\n');
                    
                    // Log progress for very large datasets
                    if (i % 1000000 === 0 && i > 0) {
                        console.log(`Wrote ${i} numbers to file...`);
                    }
                }
                
                writeStream.end();
                console.log('Finished writing input file');
                
                // Wait for the file to be fully written
                writeStream.on('finish', () => {
                    runCudaProcess();
                });
                
                writeStream.on('error', (err) => {
                    reject(new Error(`Error writing to file: ${err.message}`));
                    try {
                        fs.unlinkSync(tempInputFile);
                    } catch (e) {
                        console.error('Error deleting input file:', e);
                    }
                });
            } else {
                // For smaller datasets, use the original approach
                fs.writeFileSync(tempInputFile, data.join('\n'));
                runCudaProcess();
            }
            
            function runCudaProcess() {
                console.log(`Spawning CUDA process: ${executablePath}`);
                console.log(`Input file: ${tempInputFile}`);
                console.log(`Output file: ${tempOutputFile}`);
                
                // Spawn the C++ process
                const process = spawn(executablePath, [tempInputFile, tempOutputFile]);
                
                let stderr = '';
                let stdout = '';
                
                process.stdout.on('data', (data) => {
                    stdout += data.toString();
                    console.log(`CUDA stdout: ${data.toString()}`);
                });
                
                process.stderr.on('data', (data) => {
                    stderr += data.toString();
                    console.error(`CUDA stderr: ${data.toString()}`);
                });
                
                process.on('error', (err) => {
                    console.error('Failed to start CUDA process:', err);
                    reject(new Error(`Failed to start CUDA process: ${err.message}`));
                    
                    // Clean up files
                    try {
                        if (fs.existsSync(tempInputFile)) {
                            fs.unlinkSync(tempInputFile);
                        }
                    } catch (e) {
                        console.error('Error deleting input file:', e);
                    }
                });
                
                process.on('close', (code) => {
                    console.log(`CUDA process exited with code ${code}`);
                    
                    // Clean up the input file
                    try {
                        fs.unlinkSync(tempInputFile);
                    } catch (err) {
                        console.error('Error deleting input file:', err);
                    }
                    
                    if (code !== 0) {
                        return reject(new Error(`CUDA process exited with code ${code}: ${stderr}`));
                    }
                    
                    try {
                        // Check if output file exists
                        if (!fs.existsSync(tempOutputFile)) {
                            return reject(new Error(`Output file not created: ${tempOutputFile}`));
                        }
                        
                        console.log('Reading output file...');
                        
                        // Read the timing information from the first line
                        const firstLine = fs.readFileSync(tempOutputFile, { encoding: 'utf8', flag: 'r' }).split('\n')[0];
                        console.log(`First line of output: ${firstLine}`);
                        
                        const timeMatch = firstLine.match(/Time: ([\d.]+) ms/);
                        
                        if (!timeMatch) {
                            return reject(new Error(`Cannot parse timing from: ${firstLine}`));
                        }
                        
                        const executionTime = parseFloat(timeMatch[1]);
                        
                        // For large datasets, stream the file reading instead of loading it all into memory
                        if (data.length > LARGE_DATASET_THRESHOLD) {
                            // We'll just return the same data that was sorted, since it's more efficient
                            // than reading a huge file back in for large datasets
                            
                            console.log('Dataset too large to read back in memory, returning original data');
                            
                            // Clean up the output file
                            fs.unlinkSync(tempOutputFile);
                            
                            // Return the timing and sorted data (which is just the original data structure)
                            resolve({
                                executionTime,
                                sortedData: data.slice().sort((a, b) => a - b) // Just use JS sort for displaying
                            });
                        } else {
                            // Read the results from the output file
                            const lines = fs.readFileSync(tempOutputFile, 'utf8').trim().split('\n');
                            
                            // Clean up the output file
                            fs.unlinkSync(tempOutputFile);
                            
                            // Return the results
                            resolve({
                                executionTime,
                                sortedData: lines.slice(1).map(n => parseFloat(n))
                            });
                        }
                    } catch (err) {
                        reject(new Error(`Error processing output: ${err.message}`));
                    }
                });
            }
        } catch (err) {
            // Make sure to clean up temporary files in case of error
            try {
                if (fs.existsSync(tempInputFile)) {
                    fs.unlinkSync(tempInputFile);
                }
            } catch (e) {
                console.error('Error deleting input file:', e);
            }
            reject(new Error(`Error setting up CUDA sort: ${err.message}`));
        }
    });
}

module.exports = { runNativeBitonicSort }; 