from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ No GROQ_API_KEY found in .env")

# Initialize Groq LLM
llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")

# ✅ Use HuggingFace embeddings (free, local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample documents
docs = [
    "Python is a programming language that emphasizes readability.",
    "LangChain is a framework for building apps with LLMs.",
    "FAISS is a library for efficient similarity search on dense vectors."
]

# Create FAISS vectorstore
vectorstore = FAISS.from_texts(docs, embeddings)

# Simple array memory
history = []

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Store user input in history
    history.append(f"User: {user_input}")

    # Retrieve similar docs from FAISS
    similar_docs = vectorstore.similarity_search(user_input, k=2)
    knowledge = "\n".join([d.page_content for d in similar_docs])

    # Context = history + retrieved knowledge
    context = "\n".join(history)
    prompt = f"""Conversation so far:
{context}

Relevant knowledge:
{knowledge}

Bot:"""

    # Get response from LLM
    response = llm.invoke(prompt)

    # Add bot reply to history
    history.append(f"Bot: {response.content}")

    print("Bot:", response.content)
