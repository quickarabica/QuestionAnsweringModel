import os
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import generation_types
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
        "MATCH (e {{name: entity}})-[r:relation]-(related)
        RETURN e,r,related"
        In the cypher code make sure the arrow is towards whatever is necessary.
        give only single cypher query
        just give output for cypher as:
        Cypher code is:
        "(the code)"
        and dont include ``` this is in the answer.
        If there is NO relation found, do not generate cypher code, just print No cypher code for the above answer.
        Now, if i ask the question in whatever language, 
        give the answer in that language too
        only for cypher: entity name will be translated to 
        english only.
        """)

# Initialize conversation history
history = []

def answer_question(question):
    question = question.lower()
    
    entity1 = None
    entity2 = None

    if "relation" in question:
        parts = question.split("between")
        if len(parts) > 1:
            entities = parts[1].strip().split("and")
            if len(entities) == 2:
                entity1 = entities[0].strip()
                entity2 = entities[1].strip()
    
    if entity1 and entity2:
        result = data[(data['entity1'].str.lower() == entity1) & (data['entity2'].str.lower() == entity2)]
        
        if not result.empty:
            return result['relation'].tolist()
        else:
            return f"No relation found between {entity1} and {entity2}."
    else:
        return "Please provide a valid question format."

def generate_cypher_query(entity, relation):
    cypher_query = f"""MATCH (e {{name: '{entity}'}})-[r:{relation}]->(related)
    RETURN e, r, related"""
    return cypher_query

print("\nWelcome to the Agriculture Chatbot! Type 'exit' or 'quit' to quit.")
print("Bot: Hello!!!\n")
while True:
    user_input = input("You: ")
    
    # Check for exit condition
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break

    # Answer the question using the custom function
    model_response = answer_question(user_input)
    print(f'Question: {user_input}')
    
    # If no response, use the model to generate a response
    if isinstance(model_response, str):
        chat_session = model.start_chat(history=history)
        try:
            response = chat_session.send_message(user_input)
            model_response = response.text
        except generation_types.StopCandidateException as e:
            model_response = f"Error: {e}"

        print(f'Bot: {model_response}')
        print()
