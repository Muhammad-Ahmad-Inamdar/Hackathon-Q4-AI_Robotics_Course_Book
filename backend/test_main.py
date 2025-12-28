from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Chatbot API - Test Mode", version="1.0.0")

# Request/Response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    chapter: Optional[int] = None  # Optional chapter filter

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []
    sources_used: Optional[int] = 0
    filter_applied: Optional[str] = "None"
    context_depth: Optional[str] = "Standard"

@app.get("/")
async def root():
    return {"message": "RAG Chatbot Backend API - Test Mode Running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that processes user queries using mock RAG
    """
    try:
        # Extract user query from the last message
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break

        if not user_query:
            raise HTTPException(status_code=400, detail="No user message found in request")

        # Mock RAG processing - simulate response based on query
        # In a real implementation, this would connect to Qdrant and Gemini
        response_text = f"Thank you for your query: '{user_query}'. In the full implementation, this would be processed using RAG with Qdrant vector search and Google Gemini AI. The system would retrieve relevant context from your documents and generate an intelligent response based on the content."

        # Mock sources based on query
        sources = [f"document_{i+1}.pdf" for i in range(min(3, len(user_query.split())))]

        return ChatResponse(
            response=response_text,
            sources=sources,
            sources_used=len(sources),
            filter_applied=f"Chapter {request.chapter}" if request.chapter else "None",
            context_depth="Standard"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# Simple chat endpoint without chapter filtering
@app.post("/simple_chat", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    """
    Simple chat endpoint without chapter filtering
    """
    try:
        # Extract user query from the last message
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break

        if not user_query:
            raise HTTPException(status_code=400, detail="No user message found in request")

        # Mock processing
        response_text = f"Simple response for query: '{user_query}'. This demonstrates the basic chat functionality."
        sources = [f"source_{i+1}.pdf" for i in range(2)]

        return ChatResponse(
            response=response_text,
            sources=sources,
            sources_used=len(sources)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)