from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama model
llm = OllamaLLM(model="llama3")

# Load the dataset and create a graph
def load_and_create_graph(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 5:
                entity1, head_label, relation, entity2, tail_label = parts
                if entity1 not in graph:
                    graph[entity1] = []
                graph[entity1].append((relation, entity2))
                # Also include reverse relationships for easier searching
                if entity2 not in graph:
                    graph[entity2] = []
                graph[entity2].append((f"Reverse of {relation}", entity1))
    return graph

# Retrieve relevant chunks from the graph based on the user query
def retrieve_relevant_chunks(query, graph):
    relevant_chunks = []
    for entity in graph:
        if entity.lower() in query.lower():
            for relation, related_entity in graph[entity]:
                relevant_chunks.append(f"{entity} {relation} {related_entity}")
    return relevant_chunks

# Function to format the prompt and generate a response
def chat_with_bot(user_input, graph):
    # Retrieve relevant chunks from the graph
    relevant_chunks = retrieve_relevant_chunks(user_input, graph)

    if not relevant_chunks:
        return "Based on the dataset, I couldn't find any information related to your question."

    # Join relevant chunks to create the dataset context for the model
    dataset_context = "\n".join(relevant_chunks)

    # Define a chat prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert in analyzing CSV dataset related to agriculture, 
    the format of the csv data is (entity1,head_label,relation,entity2,tail_label)
    Relevant Data:
    {dataset_context}
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
    give only single cypher query
    just give output for cypher as:
    Cypher code is:
    "(the code)"
    and dont include ``` this is in the answer.
    If there is NO relation found, do not generate cypher code, just print No cypher code for the above answer.
    User's question: {user_input}
    Just give the direct answers, based on the prompts given above strictly!!
    
    """)

    # Format the prompt with the dataset context and user input
    formatted_prompt = prompt_template.format(user_input=user_input, dataset_context=dataset_context)

    # Get the response from the model
    response = llm.invoke(formatted_prompt)

    return response

# Load the agri dataset using the correct file path and create the graph
graph = load_and_create_graph(r"C:\Users\priya\Desktop\New folder\ranit.txt")

# Example interaction
while(True):
    user_input = input("Ask a question about agriculture: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    response = chat_with_bot(user_input, graph)
    print(response)


