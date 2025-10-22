from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from rag_pipeline import CHAIN

load_dotenv()

app = FastAPI(title="RAG Menu API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringe en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        out = CHAIN.invoke({"question": req.message})
        return ChatResponse(answer=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
