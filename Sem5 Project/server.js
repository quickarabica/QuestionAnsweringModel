const express = require('express');
const bodyParser = require('body-parser');
const neo4j = require('neo4j-driver');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());



const driver = neo4j.driver('bolt://localhost:7687', neo4j.auth.basic('neo4j', 'password'));
const session = driver.session();

// Serve the React app on the root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '/main.html')); // Serve the main index.html
});

// Endpoint to handle queries
app.post('/query', async (req, res) => {
    const question = req.body.question;

    // Convert the question to a Cypher query
    const cypherQuery = convertToCypher(question);

    try {
        const result = await session.run(cypherQuery);
        const records = result.records.map(record => record.toObject());
        res.json({ data: records });
    } catch (error) {
        console.error('Error executing Cypher query', error);
        res.status(500).send('Error executing query');
    }
});

function convertToCypher(question) {
    if (question.includes('What is the relationship between')) {
        const [entity1, entity2] = extractEntitiesFromQuestion(question);
        return `MATCH (a)-[r]->(b) WHERE a.name = '${entity1}' AND b.name = '${entity2}' RETURN a, r, b`;
    }
}

function extractEntitiesFromQuestion(question) {
    const parts = question.split(' between ')[1].split(' and ');
    return [parts[0].trim(), parts[1].trim()];
}

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
