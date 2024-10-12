const express = require('express');
const bodyParser = require('body-parser');
const neo4j = require('neo4j-driver');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Serve the main HTML page on the root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '/main.html')); // Serve the main index.html
});

app.listen(3000, () => {
    console.log('Server is running on http://localhost:3007'); // Corrected message
});
