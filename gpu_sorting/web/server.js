const express = require('express');
const path = require('path');
const app = express();
const port = process.env.PORT || 3000;
const { runNativeBitonicSort } = require('./bridge');

// Middleware to parse JSON body
app.use(express.json({ limit: '50mb' }));

// Serve static files
app.use(express.static(__dirname));

// Root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// API endpoint for native sorting
app.post('/api/sort-native', async (req, res) => {
    try {
        const { data } = req.body;
        
        if (!Array.isArray(data)) {
            return res.status(400).json({ error: 'Data must be an array of numbers' });
        }
        
        console.log(`Received request to sort ${data.length} numbers using native implementation`);
        
        const result = await runNativeBitonicSort(data);
        
        res.json({
            executionTime: result.executionTime,
            sortedData: result.sortedData
        });
    } catch (error) {
        console.error('Error in native sort:', error);
        res.status(500).json({ error: error.message });
    }
});

// Start server
app.listen(port, () => {
    console.log(`Bitonic Sort web application running at http://localhost:${port}`);
}); 