import pandas as pd
import time
from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key="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",
)

# Load the CSV data in chunks
chunks = pd.read_csv('agri.csv', chunksize=250)  # Adjust the chunk size as needed

# Prepare the prompt
def build_prompt(chunk):
    prompt = (
        "You are an expert in analyzing CSV dataset related to agriculture, "
        "the format of the csv data is (entity1,head_label,relation,entity2,tail_label). "
        "The data is given below:\n"
    )
    for _, row in chunk.iterrows():
        prompt += f"{row['entity1']}, {row['head_label']}, {row['relation']}, {row['entity2']}, {row['tail_label']}\n"
    
    prompt += (
        "Please help me answer questions related to this dataset. Ignore the "
        "tail label for those which have O. Return with all the possible answers. "
        "I will give some questions based on which extract the keywords from the question, "
        "and fetch the relation along with the answer based on the keyword. "
        "Make sure the answer is direct and exactly all answers should be given; NOTHING SHOULD BE LEFT. "
        "Take some time but scan the csv perfectly. I don't need to see relations of the entities. "
        "Make sure the answers are only from the dataset given, and if multiple matches give all. "
        "Give the answer in a paragraph manner! Make sure the answers are unique. "
        "Also make a cypher query for knowledge. Let's say you get the entity as the keyword FROM THE QUESTION, "
        "and you identify the relation; the code would be:\n"
        "MATCH (e {{name: entity}})-[r:relation]->(related)\n"
        "RETURN e,r,related\n"
        "Make sure the arrow: -> is correctly pointed every time; it may not be towards related, sometimes it can be away too. "
        "Give only single cypher query. Just give output for cypher as:\n"
        "Cypher code is:\n"
    )
    return prompt

# Loop through each chunk and ask for questions
for chunk in chunks:
    prompt = build_prompt(chunk)
    
    # Get user input for the question
    question = input("What is your question? ")
    prompt += f"Now, if I ask the question: {question}\n"

    while True:
        try:
            # Make the API call
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            # Get and print the answer
            answer = completion.choices[0].message.content
            print(f"Answer: {answer}")
            break

        except Exception as e:
            if "Rate limit reached" in str(e):
                print("Rate limit exceeded. Waiting to retry...")
                time.sleep(90)  # Wait for 90 seconds (adjust based on the error message)
            else:
                print(f"An error occurred: {e}")
                break
