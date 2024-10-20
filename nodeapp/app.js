const express = require('express');
const path = require('path'); 
const app = express();
const port = 3000;

// Middleware to parse JSON bodies
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve the main.html file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'main.html')); // Send the HTML file
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
