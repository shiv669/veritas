"""
FastAPI routes for the Veritas RAG API.

This module provides FastAPI routes to expose the Veritas RAG functionality as a REST API.
"""

import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .rag_adapter import veritas_rag_adapter
from ..config import Config
from ..mps_utils import clear_mps_cache, is_mps_available

# Create FastAPI router
router = APIRouter(prefix="/api/veritas", tags=["veritas"])

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_context_chars: Optional[int] = 3000

class QueryResponse(BaseModel):
    query: str
    context: str
    direct_answer: str
    combined_answer: str
    retrieved_chunks: List[Dict[str, Any]]

class EmbeddingRequest(BaseModel):
    text: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ConfigResponse(BaseModel):
    embedding_model: str
    llm_model: str
    device: str
    index_path: str
    mps_available: bool

# Routes
@router.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "ok", "version": "1.0.0", "name": "Veritas RAG API"}

@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get the current configuration."""
    return {
        "embedding_model": Config.EMBEDDING_MODEL,
        "llm_model": Config.LLM_MODEL,
        "device": veritas_rag_adapter.device,
        "index_path": os.path.join(Config.MODELS_DIR, "faiss"),
        "mps_available": is_mps_available()
    }

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query through the RAG system.
    
    Args:
        request: Query request
        
    Returns:
        Query results including context and answers
    """
    try:
        result = veritas_rag_adapter.query_rag(
            query=request.query,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            max_context_chars=request.max_context_chars
        )
        return result
    except Exception as e:
        # Ensure cleanup in case of error
        if veritas_rag_adapter.device == "mps":
            clear_mps_cache()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed")
async def embed(request: EmbeddingRequest):
    """
    Embed a text using the embedding model.
    
    Args:
        request: Embedding request
        
    Returns:
        Text embedding
    """
    try:
        embedding = veritas_rag_adapter.embed_query(request.text)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """
    Retrieve relevant chunks for a query.
    
    Args:
        request: Retrieve request
        
    Returns:
        List of retrieved chunks
    """
    try:
        chunks = veritas_rag_adapter.retrieve(request.query, top_k=request.top_k)
        return {
            "chunks": chunks,
            "ui_format": veritas_rag_adapter.convert_to_ui_format(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate(prompt: str = Body(..., embed=True), 
                  max_new_tokens: int = Body(200, embed=True),
                  temperature: float = Body(0.7, embed=True),
                  top_p: float = Body(0.9, embed=True)):
    """
    Generate text for a prompt.
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        
    Returns:
        Generated text
    """
    try:
        text = veritas_rag_adapter.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return {"text": text}
    except Exception as e:
        # Ensure cleanup in case of error
        if veritas_rag_adapter.device == "mps":
            clear_mps_cache()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def initialize(embedding_model: Optional[str] = None,
                    llm_model: Optional[str] = None,
                    index_path: Optional[str] = None):
    """
    Initialize the RAG system with specific models.
    
    Args:
        embedding_model: Embedding model name
        llm_model: Language model name
        index_path: FAISS index path
        
    Returns:
        Initialization status
    """
    try:
        veritas_rag_adapter.initialize(
            embedding_model=embedding_model,
            llm_model=llm_model,
            index_path=index_path
        )
        return {"status": "initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup():
    """Release resources and clean up memory."""
    try:
        veritas_rag_adapter.cleanup()
        return {"status": "cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 