"""
Veritas - A high-performance RAG system for M4 Macs

Veritas is a Retrieval-Augmented Generation (RAG) system optimized for Apple 
Silicon, particularly the M4 Mac with 128GB RAM. It provides efficient document 
retrieval and context-aware answer generation using the Mistral 2 7B model.

Core modules:
- config: Centralized system configuration
- rag: Core RAG implementation for retrieval and generation
- chunking: Document segmentation for efficient processing
- mps_utils: Apple Silicon-specific optimizations
- utils: General utility functions and logging

Usage example:
    from veritas import RAGSystem, Config, query_rag
    
    # One-time query through the RAG system
    result = query_rag("What are the key features of Veritas?")
    print(result["answer"])
    
    # Or create a persistent RAG system
    rag = RAGSystem(index_path="/path/to/index")
    context, chunks = rag.get_retrieval_context("Tell me about RAG systems")
    response = rag.generate_rag_response("How do RAG systems work?")
"""

__version__ = "0.1.0"

from .config import Config, get_device
from .chunking import chunk_text, get_chunk_size
from .rag import RAGSystem, query_rag
from .utils import setup_logging, Timer 