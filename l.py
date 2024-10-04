import os
import pandas as pd
from dotenv import load_dotenv
from llama_cpp import Llama

# Load environment variables (optional)
load_dotenv()

# Load the CSV data
data = pd.read_csv("path/to/your/csv_file.csv")

# Load LLaMA model
# Ensure the model path points to your LLaMA model (e.g., ggml model file)
model_path = "path/to/llama_model.bin"
llama_model = Llama(model_path=model_path)

# System instruction or context prompt for the model
system_instruction = f"""{data}

Please help me answer questions related to this dataset. Ignore the 
tail label for those which have 'O'. Return all the possible answers. 
I will give some questions based on which you should extract the keywords from the question, 
and fetch the relation along with the answer based on the keyword.
Make sure the answer is direct and short, and exactly all answers should be given. Take some time but scan the CSV perfectly.
I don't need to see relations of the entities.
Make sure the answers are only from the dataset given, and if multiple matches are found, give all.
Give the answer in a paragraph manner.
Make sure the answers are unique.
Also make a Cypher query for knowledge extraction. For example, if you get the entity as the keyword and you identify the relation, the code would be:
MATCH (e {{name: entity}})-[r:relation]->(related)
RETURN e,r,related
LIMIT 1
Output the Cypher code as:
Cypher code is:
(the code)
Do not include ``` in the answer.
If there is no match, do not generate Cypher code. Just print "No cypher code for the above answer."
If I ask the question in any language, 
give the answer in that language too.
For the Cypher: translate the entity name to English only.
"""

# Function to answer questions using the LLaMA model
def answer_question_llama(question):
    # Check for specific question patterns
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

    # Extract and check relation between the entities
    if entity1 and entity2:
        result = data[(data['entity1'].str.lower() == entity1) & (data['entity2'].str.lower() == entity2)]

        if not result.empty:
            return result['relation'].tolist()
        else:
            return f"No relation found between {entity1} and {entity2}."
    else:
        # Use the model to generate response if the pattern doesn't match
        response = llama_model(
            system_prompt=system_instruction,
            prompt=f"Question: {question}\n",
            max_tokens=500,
            stop=["\n", "User:"]
        )
        return response["choices"][0]["text"].strip()

# Start chat loop
print("Bot: Hello!!!\n")
while True:
    user_input = input("You: ")

    # Check for exit condition
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break

    # Get response from the model or predefined function
    model_response = answer_question_llama(user_input)
    print(f"Question: {user_input}")
    print(f'Bot: {model_response}\n')