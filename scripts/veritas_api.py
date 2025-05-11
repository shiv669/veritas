#!/usr/bin/env python3
"""
veritas_api.py

Start the Veritas RAG API server.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.veritas.api.server import run_server
from src.veritas.config import Config
from src.veritas.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)

def main():
    """Run the Veritas RAG API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Veritas RAG API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    Config.ensure_dirs()
    
    # Log server start
    logger.info(f"Starting Veritas RAG API server on {args.host}:{args.port}")
    logger.info(f"API documentation: http://{args.host}:{args.port}/docs")
    
    # Run server
    run_server(host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main() 