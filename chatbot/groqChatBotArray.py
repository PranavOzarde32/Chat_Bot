from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ No GROQ_API_KEY found in .env")

# Initialize LLM
llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")

# Simple array to store chat history
history = []

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user input to history
    history.append(f"User: {user_input}")

    # Create context from history
    context = "\n".join(history)

    # Send history + latest message to LLM
    response = llm.invoke(f"Conversation so far:\n{context}\nBot:")

    # Add bot response to history
    history.append(f"Bot: {response.content}")

    print("Bot:", response.content)
