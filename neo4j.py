import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from py2neo import Graph
from pyvis.network import Network

load_dotenv()

# Configure the API keys for both models if needed
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load CSV data
file = r"C:\Users\priya\Desktop\New folder\agri.csv"
data = pd.read_csv(file)

# Convert data to string format
multiline_string = data.to_string(index=False)

graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678")) 

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8961,
}

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
        head_label = extract_label_from_model(node1_name)
        tail_label = extract_label_from_model(node2_name)

        # Add nodes with better styles
        net.add_node(
            node1_name,
            label=node1_name,
            title=f"{node1_name} ({head_label})",
            color='#00BCD4',
            shape='circle',
            size=50,
            font={'size': 30, 'color': 'black', 'face': 'bold'},
        )

        net.add_node(
            node2_name,
            label=node2_name,
            title=f"{node2_name} ({tail_label})",
            color="#A7FFEB",
            shape='circle',
            size=10,
            font={'size': 16, 'color': 'black', 'face': 'bold'},
        )

        # Add edges with better styles
        net.add_edge(
            node1_name,
            node2_name,
            title=f"Relationship: {relationship}",
            label=relationship,
            color='black',
            arrows='to',
            length=200,
            width=2,
            font={ 'face': 'bold'}
        )
        
        # Store relationship statements
        relationship_statements.append(f"{node1_name} {relationship} {node2_name}")

    # Save the interactive graph to an HTML file
    net.save_graph('answer.html')

    # Prepare the summary of relationships
    if relationship_statements:
        return "Based on the knowledge graph, the following relationships were found:\n" + "\n".join(relationship_statements)
    return "No significant relationships were found in the knowledge graph."

labelModel = genai.GenerativeModel(
    model_name="gemini-1.5-pro",  # Change to a different model if desired
    generation_config=generation_config,
    system_instruction=f"""
        You are an expert in finding head_label and tail_label respectively, i will give you some entities, extract the corresponding label.
        give output only the label. MAKE SURE YOU GIVE THE EXACT LABEL ASSOCIATED WITH THE ENTITY.
        If you find label as 'O' just return "Other", else return the Exact label respectively. 
        Cross check to confirm the label, scan the data set thoroughly
        """
)

def extract_label_from_model(entity_name):
    # Use Model C to extract the head_label or tail_label for the given entity_name
    chat_session_c = labelModel.start_chat(history=[])
    
    # Generate the prompt for Model C to extract the label
    prompt = f"The entity is {entity_name}. Extract the corresponding label from this dataset: {multiline_string}"
    
    # Send the prompt to Model C and capture the response
    response_c = chat_session_c.send_message(prompt)
    
    # Extract the label from the response text (strip any extra spaces/newlines)
    labelModel_response = response_c.text
    
    # Handle cases where no label is found
    if labelModel_response:
        return labelModel_response
    else:
        return "Unknown"


visualize_graph("MATCH (e {name: 'apple'})-[r:Origin_Of]-(related) RETURN e,r,related")