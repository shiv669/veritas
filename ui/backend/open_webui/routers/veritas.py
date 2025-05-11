"""
Router for integrating with Veritas RAG system.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from open_webui.models.users import UserModel
from open_webui.utils.auth import get_verified_user
from open_webui.env import SRC_LOG_LEVELS
from open_webui.retrieval.models.veritas import VeritasRAGConnector

# Configure logging
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAG", logging.INFO))

# Create router
router = APIRouter(prefix="/veritas", tags=["veritas"])

# Create connector
veritas_connector = VeritasRAGConnector()

# Request and response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_context_chars: Optional[int] = 3000

class InitializeRequest(BaseModel):
    api_url: Optional[str] = "http://localhost:8000"
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    index_path: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# Routes
@router.get("/health")
async def health_check(user: UserModel = Depends(get_verified_user)):
    """Check if the Veritas RAG API is running."""
    is_healthy = veritas_connector.check_health()
    return {"status": "ok" if is_healthy else "error"}

@router.post("/initialize")
async def initialize(
    request: InitializeRequest,
    user: UserModel = Depends(get_verified_user)
):
    """Initialize the Veritas RAG system."""
    # Update API URL if provided
    if request.api_url:
        veritas_connector.api_url = request.api_url.rstrip("/")
    
    # Initialize the RAG system
    success = veritas_connector.initialize(
        embedding_model=request.embedding_model,
        llm_model=request.llm_model,
        index_path=request.index_path
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to initialize Veritas RAG")
    
    return {"status": "initialized"}

@router.get("/config")
async def get_config(user: UserModel = Depends(get_verified_user)):
    """Get the current configuration of the Veritas RAG system."""
    config = veritas_connector.get_config()
    if not config:
        raise HTTPException(status_code=500, detail="Failed to get Veritas RAG config")
    
    return config

@router.post("/query")
async def query(
    request: QueryRequest,
    user: UserModel = Depends(get_verified_user)
):
    """Process a query through the Veritas RAG system."""
    result = veritas_connector.query(
        query=request.query,
        top_k=request.top_k,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        max_context_chars=request.max_context_chars
    )
    
    return result

@router.post("/retrieve")
async def retrieve(
    request: RetrieveRequest,
    user: UserModel = Depends(get_verified_user)
):
    """Retrieve relevant chunks for a query."""
    result = veritas_connector.retrieve(
        query=request.query,
        top_k=request.top_k
    )
    
    return result

@router.post("/cleanup")
async def cleanup(user: UserModel = Depends(get_verified_user)):
    """Release resources and clean up memory."""
    success = veritas_connector.cleanup()
    
    if not success:
        log.warning("Failed to clean up Veritas RAG")
    
    return {"status": "cleaned up"} 