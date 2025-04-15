from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from chatbot import process_chat

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatInput(BaseModel):
    user_query: str
    chat_history: List[str] = []

@app.get("/")
def root():
    return {"message": "NEU Chatbot backend is running!"}

@app.post("/chat")
def chat_endpoint(payload: ChatInput):
    response = process_chat(payload.user_query, payload.chat_history)
    return {"response": response}
