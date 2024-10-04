from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Context about the dataset or topic
context_info = """
You are an expert in analyzing  dataset related to agriculture, 
    the format of the csv data is (entity1,head_label,relation,entity2,tail_label).
    
generate the cypher code also for the found answer
    
    the data is given below:
entity1,head_label,relation,entity2,tail_label
Agricultural,Agri_Process,Conjunction,horticultural,Agri_Process
yellow stem bore,Organism,Conjunction,rice leaf folder,Organism
millet crops,Crop,Includes,brown top millet,Crop
Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
storms,Natural_Disaster,Conjunction,flood,Natural_Disaster
yield-enhancing,O,Conjunction,variance-reducing,O
Multi Layer Perceptron,ML_Model,Synonym_Of,MLP,ML_Model
 
generate the cypher code also for the found answer
"""

template = """
Answer the question below
Here is the conversation history: {context}
Additional context: {context_info}
Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context": context, "context_info": context_info, "question": user_input})
        print("Bot: ", result)
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handle_conversation()
