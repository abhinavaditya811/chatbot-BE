# app/config.py

import os
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# Load API Keys from .env or environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# Set OpenAI key
openai.api_key = OPENAI_API_KEY

# Initialize SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Optional LangChain memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
