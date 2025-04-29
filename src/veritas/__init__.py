"""
Veritas - A RAG-based question answering system
"""

__version__ = "0.1.0"

from .config import Config, get_device
from .chunking import chunk_text, get_chunk_size
from .rag import RAGSystem, query_rag
from .utils import setup_logging, Timer 