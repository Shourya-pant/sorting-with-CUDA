const { spawn } = require('child_process');
const open = require('open');
const path = require('path');
const server = require('./server');

// Open browser after 2 seconds to allow server to start
setTimeout(() => {
  open('http://localhost:3000');
  console.log('Opening browser at http://localhost:3000');
}, 2000); 