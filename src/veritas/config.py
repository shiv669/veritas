"""
Main configuration settings for Veritas RAG system
"""
import os
import torch
import platform

class Config:
    """
    Configuration settings for the Veritas RAG system
    """
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
    TESTS_DIR = os.path.join(BASE_DIR, "tests")
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    
    # Input/output directories
    INPUT_DIR = os.path.join(DATA_DIR, "input")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
    INDICES_DIR = os.path.join(DATA_DIR, "indices")
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = os.path.join(MODELS_DIR, "mistral-7b")
    
    # Chunking settings
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 128
    
    # Retrieval settings
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.6
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.SCRIPTS_DIR, cls.DOCS_DIR,
                       cls.INPUT_DIR, cls.OUTPUT_DIR, cls.CHUNKS_DIR, cls.INDICES_DIR]:
            os.makedirs(dir_path, exist_ok=True)


def get_device():
    """
    Determine the best available device for computations
    Returns 'cuda', 'mps', or 'cpu' based on availability
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu" 