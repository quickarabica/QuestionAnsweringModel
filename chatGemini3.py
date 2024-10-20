import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from py2neo import Graph
from pyvis.network import Network

# Load environment variables
load_dotenv()

# Configure the API keys for both models if needed
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load CSV data
file = r"C:\Users\priya\Desktop\New folder\agri.csv"
data = pd.read_csv(file)

# Convert data to string format
dataset = data.to_string(index=False)

# Generation configurations for both models
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8961,
}

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

def generate_cypher_query(graphModel_response):
    cypher_start = "Cypher code is: "
    if cypher_start in graphModel_response:
        start_idx = graphModel_response.index(cypher_start) + len(cypher_start)
        cypher_query = graphModel_response[start_idx:]
        return cypher_query
    else:
        return None

answer=""

def visualize_graph(cypher_query):
    results = graph.run(cypher_query).data()

    # Initialize PyVis network
    net = Network(notebook=True, cdn_resources='remote')
    relationship_statements = []

    for result in results:
        node1_name = result['e']['name']
        node2_name = result['related']['name']
        relationship = result['r'].__class__.__name__

        # Extract labels for nodes
        head_label = extract_label_from_e1(node1_name)
        tail_label = extract_label_from_e2(node2_name)

        # Add nodes with better styles
        net.add_node(
            node1_name,
            label=node1_name,
            title=f"{node1_name} ({head_label})",
            color='#00BCD4',
            shape='circle',
            size=30,
            font={'size': 20, 'color': 'black', 'face': 'bold'},
        )

        net.add_node(
            node2_name,
            label=node2_name,
            title=f"{node2_name} ({tail_label})",
            color="#A7FFEB",
            shape='circle',
            size=10,
            font={'size': 16, 'color': 'black'},
        )

        # Add edges with better styles
        net.add_edge(
            node1_name,
            node2_name,
            title=f"Relationship: {relationship}",
            label=relationship,
            color='black',
            arrows='to',
            length=150,
            width=2,
            font={ 'face': 'bold'},
            physics=False
        )
        

        # Store relationship statements
        relationship_statements.append(f"{node1_name} {relationship} {node2_name}")

    # Save the interactive graph to an HTML file
    net.save_graph('answer.html')

    # Prepare the summary of relationships
    if relationship_statements:
        return "Based on the knowledge graph, the following relationships were found:\n" + "\n".join(relationship_statements)
    return "No significant relationships were found in the knowledge graph."

# Initialize Model A (for Cypher query generation)

graphModel = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction=f"""
        You are an expert in analyzing a CSV dataset related to agriculture.
        The format of the CSV data is (entity1, head_label, relation, entity2, tail_label).
        The data is given below:
        {dataset}
        Your task is to generate a Cypher query based on the entity and relation from the user's question.
        the code would be stricly this:
        MATCH (e {{name: entity}})-[r:relation]-(related) RETURN e,r,related
        - Make sure the entity name and relation match exactly, both in terms of case sensitivity and spelling with the dataset.
        - If there is NO relation found, do not generate Cypher code, just print: No Cypher code for the above answer.
        The Cypher code should be output as:
        Cypher code is: (the code)
        and dont include ``` this is in the answer.
        If there is NO relation found, do not generate cypher code, just print No cypher code for the above answer.
        """
        )

# Initialize Model B (for answering questions or summarizing relationships)
summaryModel = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # Change to a different model if desired
    generation_config=generation_config,
    system_instruction=f"""
        You are an expert in summarizing relationships between entities.
        Based on the following relationships: {answer}, provide a summary in one or two sentences.
        """
)

def extract_label_from_e1(entity_name):
    start = entity_name
    if start in dataset:
        # Find the start index of the entity_name
        start_idx = dataset.index(start) + len(start)
        
        # Search for the next meaningful word after spaces
        after_entity = dataset[start_idx:].strip()
        head_label = after_entity.split()[0]  # The first word after the entity_name
        
        return head_label

def extract_label_from_e2(entity_name):
    start = entity_name
    if start in dataset:
        # Find the start index of the entity_name
        start_idx = dataset.index(start) + len(start)
        
        # Search for the next meaningful word after spaces
        after_entity = dataset[start_idx:].strip()
        tail_label = after_entity.split()[0]  # The first word after the entity_name
    
        return tail_label

def answer_question_with_two_models(question):
    # Step 1: Use Model A to generate Cypher query
    chat_session_a = graphModel.start_chat(history=[])
    response_a = chat_session_a.send_message(question)
    graphModel_response = response_a.text
    
    # Generate Cypher query
    query = generate_cypher_query(graphModel_response)
    graph_summary = visualize_graph(query)
    
    # Step 2: Use Model B to summarize the graph relationships
    chat_session_b = summaryModel.start_chat(history=[])
    response_b = chat_session_b.send_message(graph_summary)
    summaryModel_response = response_b.text
    
    # Output the final summarized answer
    print(f"Bot: {summaryModel_response}{graphModel_response} ")
    print("Interactive graph generated and saved as 'answer.html'.\n")


# Chatbot interaction
print("\nWelcome to the Agriculture Chatbot! Type 'exit' or 'quit' to quit.")
print("Bot: Hello!!!\n")
while True:
    user_input = input("You: ")

    # Check for exit condition
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break

    # Answer using two models
    answer_question_with_two_models(user_input)
