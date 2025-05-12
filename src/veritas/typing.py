"""
Veritas Type Definitions

This module contains common type definitions used throughout the Veritas project.
These types provide consistent typing across the codebase and improve static analysis.
"""

import os
import numpy as np
from typing import Dict, List, Any, Union, Optional, TypeVar, Tuple, Callable, Protocol

# Basic types
PathLike = Union[str, os.PathLike]
JSON = Dict[str, Any]

# RAG system types
ChunkType = Dict[str, Any]
ChunkList = List[ChunkType]
QueryType = str
EmbeddingType = Union[List[float], np.ndarray]
RetrievalResult = Dict[str, Any]
ModelOutput = Dict[str, Any]

# Model types
ModelType = TypeVar('ModelType')
TokenizerType = TypeVar('TokenizerType')

# Message types for chat interfaces
MessageRole = str  # "system", "user", "assistant"
MessageContent = str
Message = Dict[str, Union[MessageRole, MessageContent]]
MessageList = List[Message]

# AI Scientist types
ResearchIdea = Dict[str, Any]
ResearchTemplate = Dict[str, Any]
ExperimentConfig = Dict[str, Any]
EvaluationMetric = Dict[str, Union[str, float]]

# Callback types
ProgressCallback = Callable[[int, int, str], None]

# Protocol for LLM interfaces
class LLMInterface(Protocol):
    """Protocol defining the interface for language model interactions."""
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        ...
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text into a vector representation."""
        ...

# Health check types
HealthStatus = Dict[str, Union[bool, str, Dict[str, Any]]] 