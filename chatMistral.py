from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API"))

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

file = r"C:\Users\ranit\OneDrive\Desktop\git\agri.csv"
data = pd.read_csv(file)

multiline_string = data.to_string(index=False)

def search_dataset(query):
    df_filtered = data[
        (data['entity1'].str.contains(query, case=False)) |
        (data['entity2'].str.contains(query, case=False)) |
        (data['head_label'].str.contains(query, case=False)) |
        (data['tail_label'].str.contains(query, case=False))
    ]
    return df_filtered

def generate_response(question):
    dataset_result = search_dataset(question)
    
    if not dataset_result.empty:
        answers = []
        for _, row in dataset_result.iterrows():
            answers.append(f"{row['entity1']} has a relation with {row['entity2']} through {row['relation']}.")
        response = " ".join(answers)
    else:
        prompt = f"""
        You are an expert in analyzing CSV datasets related to agriculture. 
        The format of the data is (entity1, head_label, relation, entity2, tail_label). 
        The dataset is below:\n{multiline_string}\n
        Scan the data thoroughly and answer questions related to this dataset.
        Question: {question}
        """
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def generate_cypher_query(entity, relation):
    cypher_query = f"MATCH (e {{name: '{entity}'}})-[r:{relation}]->(related) RETURN e, r, related"
    return cypher_query

def chat():
    print("Bot: Hello! I'm your agriculture expert chatbot.\n")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        
        response = generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
