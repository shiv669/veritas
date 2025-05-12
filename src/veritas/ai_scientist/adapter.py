"""
Veritas AI Scientist - LLM Interface Module

This module provides the interface for the Veritas AI Scientist
to work with Mistral models via the RAG system.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import gc
import torch

# Import from Veritas
from src.veritas.rag import RAGSystem
from src.veritas.config import Config, get_device
from src.veritas.mps_utils import clear_mps_cache

logger = logging.getLogger(__name__)

class ResponseMessage:
    """Simple class for response message objects"""
    def __init__(self, content):
        self.content = content

class ResponseChoice:
    """Simple class for response choice objects"""
    def __init__(self, message_content):
        self.message = ResponseMessage(message_content)

class LLMResponse:
    """Simple class for LLM response objects"""
    def __init__(self, content):
        self.choices = [ResponseChoice(content)]

class MistralAdapter:
    """
    Adapter for the Mistral model with standardized interface.
    
    This adapter provides a consistent interface for working with
    the Mistral RAG system.
    """
    def __init__(self, rag_system=None):
        """
        Initialize the adapter with a RAGSystem instance.
        
        Args:
            rag_system: Optional RAGSystem instance. If None, a new one will be created.
        """
        self.rag_system = rag_system
        if self.rag_system is None:
            logger.info("Creating new RAGSystem instance")
            self.rag_system = RAGSystem(
                embedding_model=Config.EMBEDDING_MODEL,
                llm_model=Config.LLM_MODEL,
                device=get_device()
            )
        self.memory_counter = 0  # Counter for managing memory cleanup
    
    def _clean_memory_if_needed(self):
        """Clean up memory periodically to prevent OOM errors"""
        self.memory_counter += 1
        if self.memory_counter % 10 == 0:  # Every 10 calls
            logger.info("Cleaning memory")
            gc.collect()
            if self.rag_system.device == "mps":
                clear_mps_cache()
    
    def _extract_latest_message(self, messages: List[Dict[str, str]]) -> str:
        """
        Extract the latest user message from a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The content of the latest user message
        """
        # Find the last user message in the conversation
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]
        
        # Fallback: concatenate all messages as context
        return " ".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                        for msg in messages])
    
    def _extract_system_message(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract the system message from a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The content of the system message, or None if no system message
        """
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                return msg["content"]
        return None
    
    def generate_chat_completion(self, 
                               model: str, 
                               messages: List[Dict[str, str]], 
                               temperature: float = 0.7,
                               max_tokens: int = 500,
                               n: int = 1,
                               **kwargs) -> LLMResponse:
        """
        Generate chat completions with standardized format.
        
        Args:
            model: Model name (uses Mistral)
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Max tokens to generate
            n: Number of completions (only 1 supported)
            **kwargs: Additional args
            
        Returns:
            A response object with the generated content
        """
        # Extract the query from messages
        query = self._extract_latest_message(messages)
        system_msg = self._extract_system_message(messages)
        
        # Combine system message with query if exists
        full_query = f"{system_msg}\n\n{query}" if system_msg else query
        
        # Generate response
        result = self.rag_system.generate_rag_response(
            query=full_query,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=kwargs.get('top_k', 5),
            max_context_chars=kwargs.get('max_context_chars', 3000)
        )
        
        # Clean memory if needed
        self._clean_memory_if_needed()
        
        # Return in standard format
        return LLMResponse(result["combined_response"])
    
    # For backward compatibility
    chat_completions_create = generate_chat_completion
    
    def generate_message(self, 
                       model: str,
                       messages: List[Dict[str, Union[str, List[Dict[str, str]]]]], 
                       system: str = None,
                       max_tokens: int = 500,
                       temperature: float = 0.7,
                       **kwargs) -> LLMResponse:
        """
        Generate messages with standardized format.
        
        Args:
            model: Model name (uses Mistral)
            messages: List of message format dicts
            system: System message
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional args
            
        Returns:
            A response object with the generated content
        """
        # Extract the query from message format
        query = ""
        
        # Handle complex message format
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                # Content can be list of blocks or simple string
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            query = block.get("text", "")
                            break
                else:
                    query = content
                break
        
        # Prepend system message if provided
        if system:
            query = f"{system}\n\n{query}"
        
        # Generate response
        result = self.rag_system.generate_rag_response(
            query=query,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=kwargs.get('top_k', 5),
            max_context_chars=kwargs.get('max_context_chars', 3000)
        )
        
        # Clean memory if needed
        self._clean_memory_if_needed()
        
        # Return in standard format
        return LLMResponse(result["combined_response"])
    
    # For backward compatibility
    messages_create = generate_message


def create_mistral_client(model_name=None):
    """
    Create a Mistral client for the Veritas AI Scientist.
    
    Args:
        model_name: Model name (optional)
        
    Returns:
        MistralAdapter instance
    """
    logger.info(f"Creating Mistral client (model_name={model_name})")
    return MistralAdapter() 