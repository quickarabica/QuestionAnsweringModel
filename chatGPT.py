import openai
openai.api_key='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'
messages = [ {"role": "system", "content":"You are a intelligent assistant."} ]
while True:
   message = "Raj"
   if message:
      messages.append(
         {"role": "user", "content": message},
      )
      chat = openai.chat.completions.create(
         model="gpt-3.5", messages=messages
      )
   answer = chat.choices[0].message.content
   print(f"ChatGPT: {answer}")
   messages.append({"role": "assistant", "content": answer}) 