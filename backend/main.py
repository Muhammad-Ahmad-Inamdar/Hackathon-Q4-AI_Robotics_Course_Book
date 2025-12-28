from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
from src.rag_agent.agent import RAGAgent

load_dotenv()

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent
try:
    rag_agent = RAGAgent()
    rag_agent_available = True
    print("✅ RAG Agent Initialized")
except Exception as e:
    print(f"❌ RAG Agent Init Failed: {e}")
    rag_agent_available = False


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    chapter: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    sources_used: int = 0
    filter_applied: str = "None"
    context_depth: str = "Standard"


@app.get("/")
async def root():
    return {"status": "online", "agent_ready": rag_agent_available}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not rag_agent_available:
        return ChatResponse(response="Agent offline.")

    user_query = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        None
    )

    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    result = rag_agent.process_query(user_query, request.chapter)

    return ChatResponse(
        response=result["answer"],
        sources=[f"Source Chunk {i+1}" for i in range(result["sources_used"])],
        sources_used=result["sources_used"],
        filter_applied=result["filter_applied"],
        context_depth=result["context_depth"]
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
