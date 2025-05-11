"""
API package for Veritas RAG

This package provides a REST API for the Veritas RAG system.
"""

from .rag_adapter import veritas_rag_adapter
from .routes import router
from .server import create_app, run_server

__all__ = ['veritas_rag_adapter', 'router', 'create_app', 'run_server'] 