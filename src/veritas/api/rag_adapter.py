"""
RAG API adapter for Veritas

This module provides adapter functions to expose the Veritas RAG System to the UI.
It bridges the gap between the UI's expected inputs/outputs and our optimized RAGSystem.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import faiss
import numpy as np
from pathlib import Path

from ..config import Config, get_device
from ..rag import RAGSystem
from ..mps_utils import clear_mps_cache, is_mps_available
from ..utils import setup_logging

logger = setup_logging(__name__)

class VeritasRAGAdapter:
    """
    Adapter class for exposing the Veritas RAG System to external interfaces.
    
    This adapter provides methods that match the expected interfaces of the UI,
    delegating to the optimized RAGSystem implementation internally.
    """
    
    def __init__(self):
        """Initialize the adapter without loading models to conserve memory."""
        self.rag_system = None
        self.device = get_device()
        self.initialized = False
    
    def initialize(self, 
                  embedding_model: str = None,
                  llm_model: str = None,
                  index_path: str = None) -> None:
        """
        Initialize the RAG system with models.
        
        Args:
            embedding_model: Embedding model name or path
            llm_model: Language model name or path
            index_path: Path to FAISS index directory
        """
        try:
            # Ensure directories exist
            Config.ensure_dirs()
            
            # Create default index path if not provided
            if not index_path:
                index_path = os.path.join(Config.MODELS_DIR, "faiss")
            
            # Initialize the RAG system
            self.rag_system = RAGSystem(
                embedding_model=embedding_model or Config.EMBEDDING_MODEL,
                llm_model=llm_model or Config.LLM_MODEL,
                index_path=index_path,
                device=self.device
            )
            
            self.initialized = True
            logger.info("VeritasRAGAdapter initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VeritasRAGAdapter: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def ensure_initialized(self):
        """Ensure the RAG system is initialized."""
        if not self.initialized or not self.rag_system:
            logger.info("Initializing RAG system on first use")
            self.initialize()
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query using the embedding model.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding
        """
        self.ensure_initialized()
        return self.rag_system.embed_query(query)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        self.ensure_initialized()
        return self.rag_system.retrieve(query, top_k=top_k)
    
    def generate(self, 
                prompt: str, 
                max_new_tokens: int = 200,
                temperature: float = 0.7,
                top_p: float = 0.9) -> str:
        """
        Generate text for a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated text
        """
        self.ensure_initialized()
        
        # Clear MPS cache before generation if using Apple Silicon
        if self.device == "mps":
            clear_mps_cache()
        
        response = self.rag_system.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Clear MPS cache after generation if using Apple Silicon
        if self.device == "mps":
            clear_mps_cache()
        
        return response
    
    def query_rag(self, 
                 query: str, 
                 top_k: int = 5,
                 max_new_tokens: int = 200,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_context_chars: int = 3000) -> Dict[str, Any]:
        """
        Process a query through the RAG system.
        
        Args:
            query: The user's query
            top_k: Number of chunks to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            max_context_chars: Maximum characters in the context
            
        Returns:
            Dictionary with query results including context and answers
        """
        self.ensure_initialized()
        
        # Use the RAGSystem's generate_rag_response method
        result = self.rag_system.generate_rag_response(
            query=query,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_context_chars=max_context_chars
        )
        
        # Convert to UI-compatible format
        ui_result = {
            "query": query,
            "retrieved_chunks": result["retrieved_chunks"],
            "context": result["context"],
            "direct_answer": result["direct_response"],
            "combined_answer": result["combined_response"],
            "answer": result["combined_response"]  # For compatibility with older code
        }
        
        return ui_result
    
    def convert_to_ui_format(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert retrieved chunks to UI-compatible format.
        
        Args:
            retrieved_chunks: List of retrieved chunks from RAGSystem
            
        Returns:
            Dictionary with UI-compatible format
        """
        distances = [chunk["score"] for chunk in retrieved_chunks]
        documents = [chunk["chunk"] for chunk in retrieved_chunks]
        metadatas = [{"score": chunk["score"], "index": chunk["index"]} for chunk in retrieved_chunks]
        
        return {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas]
        }
    
    def cleanup(self):
        """Release resources and clean up memory."""
        if self.rag_system:
            # Release models
            self.rag_system.model = None
            self.rag_system.tokenizer = None
            self.rag_system.generator = None
            self.rag_system.embedding_model = None
            
            # Release FAISS index
            self.rag_system.index = None
            self.rag_system.chunks = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear MPS cache if using Apple Silicon
            if self.device == "mps":
                clear_mps_cache()
        
        self.initialized = False
        logger.info("VeritasRAGAdapter resources released")

# Create a singleton instance for use throughout the application
veritas_rag_adapter = VeritasRAGAdapter() 