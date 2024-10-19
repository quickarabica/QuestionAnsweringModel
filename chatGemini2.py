import os
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import generation_types
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from py2neo import Graph
import graphviz

# Load environment variables
load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load CSV data
file = r"C:\Users\priya\Desktop\New folder\agri.csv"
data = pd.read_csv(file)

# Convert to string format (if necessary for processing or embedding into the model)
multiline_string = data.to_string(index=False)

# Create the model with safer configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 100000,  # Limit output tokens
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    system_instruction=f"""
    You are an expert in analyzing CSV dataset related to agriculture, 
        the format of the csv data is (entity1,head_label,relation,entity2,tail_label)
        the data is given below:
        {multiline_string}
        Scan the data thoroughly and do not leave the answers incomplete, check more than twice before giving answers.
        Please help me answer questions related to this dataset. Return with ALL the possible answers. 
        i will give some questions based on which extract the keywords from the question, 
        and fetch the relation along with the answer based on the keyword.
        make sure the answer is direct and exactly all answers should be given NOTHING SHOULD BE LEFT, 
        take some time but scan the data perfectly.
        i dont need to see relations of the entities.
        make sure the answers are only from dataset given and if multiple matches give all.
        Give the answer in a paragraph manner!
        Make sure the answers are unique.
        Also make a cypher query for knowledge.
        lets say you get the entity as the keyword FROM THE QUESTION and you identify the relation, the code would be stricly this:
        MATCH (e {{name: entity}})-[r:relation]-(related) RETURN e,r,related
        give only single cypher query
        just give output for cypher as:
        Cypher code is: (the code)
        and dont include ``` this is in the answer.
        If there is NO relation found, do not generate cypher code, just print No cypher code for the above answer.
        Now, if i ask the question in whatever language, 
        give the answer in that language too
        only for cypher: entity name will be translated to 
        english only.
"""
)
    
# Initialize conversation history
history = []

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

def generate_cypher_query(model_response):
    cypher_start = "Cypher code is: "
    if cypher_start in model_response:
        start_idx = model_response.index(cypher_start) + len(cypher_start)
        cypher_query = model_response[start_idx:]
        return cypher_query
    else:
        return None

def visualize_graph(cypher_query):
    results = graph.run(cypher_query).data()
    
    dot = graphviz.Digraph(comment='Neo4j Graph')

    for result in results:
        node1_name = result['e']['name']
        node2_name = result['related']['name']
        relationship = result['r'].__class__.__name__
        
        dot.node(node1_name, node1_name)
        dot.node(node2_name, node2_name)
        dot.edge(node1_name, node2_name, label=relationship)

    # Save the visualization as PNG
    dot.render('neo4j_graph', format='png')
    print("Graph generated and saved as 'neo4j_graph.png'.")

def answer_question(question):
    # Send question to Gemini AI model
    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(question)
    
    # Extract the response text
    model_response = response.text
    print(f'Bot: {model_response}\n')

    # Extract Cypher query if available and visualize
    cypher_query = generate_cypher_query(model_response)
    visualize_graph(cypher_query)
    return model_response

# Chatbot interaction
print("\nWelcome to the Agriculture Chatbot! Type 'exit' or 'quit' to quit.")
print("Bot: Hello!!!\n")
while True:
    user_input = input("You: ")
    
    # Check for exit condition
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break

    # Answer the question and handle response
    answer_question(user_input)

