"""
Main configuration settings for Veritas RAG system

This module defines the global configuration settings for the Veritas Retrieval-Augmented 
Generation (RAG) system. It centralizes all parameters in a single location to improve 
maintainability and make system-wide changes easier to implement.

Features:
- Centralized configuration via the Config class
- Dynamic path configurations for file storage
- Model selection parameters
- Performance optimization settings for Apple Silicon
- Environment variable management
- Device selection utilities

Usage examples:
    from veritas.config import Config, get_device
    
    # Access a configuration parameter
    models_dir = Config.MODELS_DIR
    
    # Create necessary directories
    Config.ensure_dirs()
    
    # Set up optimized environment variables
    Config.setup_environment()
    
    # Get the best available computation device
    device = get_device()
"""
import os
import torch
import platform
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Centralized configuration class for the Veritas RAG system.
    
    This class provides a single point of control for all system settings
    and parameters. It includes file paths, model configurations, chunking
    parameters, and performance optimization settings.
    
    Attributes:
        BASE_DIR (str): The root directory of the project
        DATA_DIR (str): Directory for storing data files
        MODELS_DIR (str): Directory for storing model files
        LOGS_DIR (str): Directory for storing log files
        SCRIPTS_DIR (str): Directory for storing script files
        TESTS_DIR (str): Directory for storing test files
        DOCS_DIR (str): Directory for storing documentation
        INPUT_DIR (str): Directory for input files requiring processing
        OUTPUT_DIR (str): Directory for processed output files
        CHUNKS_DIR (str): Directory for storing text chunks
        INDICES_DIR (str): Directory for storing FAISS indices
        TEMP_DIR (str): Temporary directory for intermediate files (on SSD)
        EMBEDDING_MODEL (str): Path or name of the embedding model
        LLM_MODEL (str): Path or name of the language model
        DEFAULT_CHUNK_SIZE (int): Default size for text chunks (in words)
        DEFAULT_CHUNK_OVERLAP (int): Default overlap between chunks (in words)
        TOP_K (int): Number of chunks to retrieve for each query
        SIMILARITY_THRESHOLD (float): Minimum similarity for relevant chunks
        CPU_THREADS (int): Number of CPU threads to use
        MEMORY_LIMIT_GB (int): Maximum memory limit in gigabytes
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
    INDICES_DIR = os.path.join(MODELS_DIR, "faiss")
    
    # Temporary directory for files - Uses SSD by default
    TEMP_DIR = "/Volumes/8SSD/veritas/tmp"
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Converts text to vectors
    LLM_MODEL = os.path.join(MODELS_DIR, "mistral-7b")  # The AI that generates answers
    
    # Chunking settings
    DEFAULT_CHUNK_SIZE = 512  # How many words in each text chunk
    DEFAULT_CHUNK_OVERLAP = 128  # How much chunks overlap to avoid splitting ideas
    
    # Retrieval settings
    TOP_K = 5  # How many relevant chunks to retrieve for each question
    SIMILARITY_THRESHOLD = 0.6  # How similar a chunk must be to be considered relevant
    
    # M4 Mac performance settings
    CPU_THREADS = 4
    MEMORY_LIMIT_GB = 100  # Max memory limit in GB
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.SCRIPTS_DIR, cls.DOCS_DIR,
                       cls.INPUT_DIR, cls.OUTPUT_DIR, cls.CHUNKS_DIR, cls.INDICES_DIR, cls.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def setup_environment(cls):
        """
        Set up all environment variables in one centralized location
        
        This function sets all the environment variables needed for optimal
        performance on M4 Mac with 128GB RAM and using an SSD for storage.
        """
        logger.info(f"Setting up environment variables with temp dir: {cls.TEMP_DIR}")
        
        # Performance optimization environment variables
        env_vars = {
            # PyTorch/MPS optimizations
            'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable upper limit to prevent OOM
            'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.6',  # Keep more in memory
            'PYTORCH_MPS_MEMORY_LIMIT': f'{cls.MEMORY_LIMIT_GB}GB',  # Use available RAM
            'PYTORCH_ENABLE_MPS_FALLBACK': '1',  # Enable fallback
            
            # Thread/CPU optimizations
            'OMP_NUM_THREADS': str(cls.CPU_THREADS),  # OpenMP threads
            'MKL_NUM_THREADS': str(cls.CPU_THREADS),  # MKL threads
            'NUMEXPR_NUM_THREADS': str(cls.CPU_THREADS),  # NumExpr threads
            'TOKENIZERS_PARALLELISM': 'true',  # Enable tokenizer parallelism for speed
            
            # Temporary directory settings (use SSD)
            'TMPDIR': cls.TEMP_DIR,  # Main temp directory
            'TORCH_HOME': os.path.join(cls.TEMP_DIR, 'torch'),  # PyTorch cache
            'TRANSFORMERS_CACHE': os.path.join(cls.TEMP_DIR, 'transformers'),  # Transformers cache
            'HF_HOME': os.path.join(cls.TEMP_DIR, 'huggingface'),  # HuggingFace cache
            
            # Misc optimizations
            'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',  # Reduce warnings
        }
        
        # Apply all environment variables
        os.environ.update(env_vars)
        logger.info(f"Set {len(env_vars)} environment variables for optimal performance")
        
        return env_vars


def get_device():
    """
    Determine the best available device for computations
    
    This version focuses exclusively on Apple Silicon (MPS) support
    with fallback to CPU if MPS is not available.
    
    Returns:
        'mps' if Apple Silicon is available, otherwise 'cpu'
    """
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon) for computation")
        return "mps"
    else:
        logger.info("MPS not available, using CPU for computation")
        return "cpu" 