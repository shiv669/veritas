"""
API server for Veritas RAG

This module provides a FastAPI server to expose the Veritas RAG functionality as a REST API.
"""

import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

from .routes import router
from ..config import Config, get_device
from ..mps_utils import optimize_memory_for_m4
from ..utils import setup_logging

# Configure logging
logger = setup_logging(__name__)

def create_app():
    """Create and configure the FastAPI application."""
    # Apply memory optimizations for M4 Mac
    optimize_memory_for_m4()
    
    # Create FastAPI app
    app = FastAPI(
        title="Veritas RAG API",
        description="API for Retrieval-Augmented Generation using Veritas",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    @app.get("/")
    def root():
        """Root endpoint that returns basic info about the API."""
        return {
            "name": "Veritas RAG API",
            "version": "1.0.0",
            "documentation": "/docs",
            "device": get_device()
        }
    
    @app.on_event("startup")
    async def startup_event():
        """Initialization on server startup."""
        Config.ensure_dirs()
        logger.info(f"Veritas RAG API starting on device: {get_device()}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on server shutdown."""
        from .rag_adapter import veritas_rag_adapter
        veritas_rag_adapter.cleanup()
        logger.info("Veritas RAG API shutting down, resources released")
    
    return app

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Veritas RAG API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, reload=args.reload) 