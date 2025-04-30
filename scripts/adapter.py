#!/usr/bin/env python3
"""
adapter.py - Connect Veritas RAG to Open WebUI
"""
import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.veritas.rag import RAGSystem
from src.veritas.config import Config

# Initialize the FastAPI app
app = FastAPI(title="Veritas RAG API")

# Initialize RAG system with the latest index
def get_latest_index():
    import re
    index_dirs = [d for d in os.listdir(Config.INDICES_DIR) 
                 if os.path.isdir(os.path.join(Config.INDICES_DIR, d))]
    ts_pattern = re.compile(r'^\d{8}_\d{6}$')
    ts_dirs = [d for d in index_dirs if ts_pattern.match(d)]
    if ts_dirs:
        return os.path.join(Config.INDICES_DIR, sorted(ts_dirs)[-1])
    else:
        return os.path.join(Config.INDICES_DIR, sorted(index_dirs)[-1])

# Initialize the RAG system
print("Initializing RAG system...")
latest_index = get_latest_index()
print(f"Using index at: {latest_index}")
rag_system = RAGSystem(index_path=latest_index)
print("RAG system initialized.")

# Define request model
class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.7
    top_k: int = 3
    max_tokens: int = 1024

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    # Extract the last user message
    last_message = request.messages[-1]["content"]
    print(f"Received question: {last_message}")
    
    # Retrieve relevant context
    retrieved_chunks = rag_system.retrieve(last_message, top_k=request.top_k)
    print(f"Retrieved {len(retrieved_chunks)} chunks")
    
    # Format context for RAG
    context = "\n\n".join([result['chunk'].get('text', '') for result in retrieved_chunks])
    
    # Build prompt with context
    prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {last_message}

Answer:"""
    
    # Generate response
    response = rag_system.generate(
        prompt, 
        temperature=request.temperature,
        max_length=request.max_tokens
    )
    print(f"Generated response of length {len(response)}")
    
    # Format the response for the UI
    return {
        "id": "rag-response",
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": "veritas-rag",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(response),
            "total_tokens": len(prompt) + len(response)
        },
        "sources": [
            {
                "id": str(i),
                "text": chunk['chunk'].get('text', '')[:500],
                "score": float(chunk['score'])
            } for i, chunk in enumerate(retrieved_chunks)
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_system": "initialized"}

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Run the server if executed directly
if __name__ == "__main__":
    print("Starting Veritas RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 