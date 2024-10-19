from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

file = r"C:\Users\priya\Desktop\New folder\agri.csv"

data = pd.read_csv(file)

# Convert to string format (if necessary for processing or embedding into the model)
multiline_string = data.to_string(index=False)
# Context about the dataset or topic
context_info = f"""
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
        MATCH (e {{name: entity}})-[r:relation]-(related)
        RETURN e,r,related
        give only single cypher query
        just give output for cypher as:
        Cypher code is:
        "(the code)"
        and dont include ``` this is in the answer.
        If there is NO relation found, do not generate cypher code, just print No cypher code for the above answer.
        Now, if i ask the question in whatever language, 
        give the answer in that language too
        only for cypher: entity name will be translated to 
        english only."""
template = """
Answer the question below
Here is the conversation history: {context}
Additional context: {context_info}
Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3.2")
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
