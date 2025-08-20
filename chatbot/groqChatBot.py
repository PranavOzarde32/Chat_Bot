from dotenv import load_dotenv
import os

load_dotenv()  # this reads .env
api_key = os.getenv("GROQ_API_KEY")

if api_key is None:
    raise ValueError("❌ No GROQ_API_KEY found. Check your .env or environment variables.")

print(f"Using API key: {api_key[:8]}...")  # safe now

from langchain_groq import ChatGroq
llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")
print("🤖 Chatbot ready! Type 'exit' or 'quit' to stop.\n")

# Chat loop
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = llm.invoke(user_input)
        print("Bot:", response.content)

    except Exception as e:
        print(f"⚠️ Error: {e}")
