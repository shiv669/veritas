"""
Veritas AI Scientist - AI-powered Scientific Research Assistant

This package provides tools and modules for generating research ideas,
designing experiments, and producing scientific writeups using the
Mistral model with RAG capabilities.
"""

from .adapter import MistralAdapter, create_mistral_client
from .memory_manager import clean_memory, memory_check_decorator
from .prompt_strategy import get_optimized_prompt, format_prompt_for_mistral

__version__ = "0.2.0"

__all__ = [
    "MistralAdapter",
    "create_mistral_client",
    "clean_memory",
    "memory_check_decorator",
    "get_optimized_prompt",
    "format_prompt_for_mistral",
] 