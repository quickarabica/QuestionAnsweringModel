from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

data = [
    {"entity1": "Agricultural", "head_label": "Agri_Process", "relation": "Conjunction", "entity2": "horticultural", "tail_label": "Agri_Process"},
    {"entity1": "yellow stem bore", "head_label": "Organism", "relation": "Conjunction", "entity2": "rice leaf folder", "tail_label": "Organism"},
    {"entity1": "millet crops", "head_label": "Crop", "relation": "Includes", "entity2": "brown top millet", "tail_label": "Crop"}
]

df = pd.DataFrame(data)

def search_dataset(query):
    df_filtered = df[df['tail_label'] != 'O']
    result = df_filtered[
        (df_filtered['entity1'].str.contains(query, case=False)) |
        (df_filtered['entity2'].str.contains(query, case=False)) |
        (df_filtered['head_label'].str.contains(query, case=False)) |
        (df_filtered['tail_label'].str.contains(query, case=False))
    ]
    return result

def generate_response(prompt):
    instruction = """
    Answer questions related to this dataset. Ignore the tail label for those which have 'O'. 
    Return all possible answers. Extract the keywords from the question, and fetch the relation along with the answer 
    based on the keyword. Make sure the answer is direct and complete. Scan the dataset perfectly, and give all answers 
    without leaving anything. The answer should be in paragraph form and unique.
    """
    dataset_result = search_dataset(prompt)
    if not dataset_result.empty:
        answers = []
        for index, row in dataset_result.iterrows():
            answers.append(f"{row['entity1']} has a relationship with {row['entity2']}.")
        response = " ".join(set(answers))
        return response
    else:
        inputs = tokenizer(instruction + prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def chat():
    print("Welcome to the Agriculture Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
