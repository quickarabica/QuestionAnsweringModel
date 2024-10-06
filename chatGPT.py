import openai
openai.api_key='sk-svcacct-IYG-92_kn0qkjVv9X9Efc8WCQbTccuPwD78HlquBZdRuE-Lct7iKICnlBkyT3BlbkFJISoWW9ABQ2kGhzgJcoB1GxfmjJIcE3fzoI4BFXG_RXI1FuQlbs-2dZMPyzgA'
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