"""
Veritas - A RAG (Retrieval-Augmented Generation) system for efficient document processing and question answering.
"""

from veritas.rag import RAGSystem
from veritas.config import (
    # Paths
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    SCRIPTS_DIR,
    
    # Data files
    INPUT_DATA_FILE,
    RAG_CHUNKS_FILE,
    
    # Model files
    FAISS_INDEX_FILE,
    METADATA_FILE,
    MISTRAL_MODEL_DIR,
    MISTRAL_FAISS_INDEX,
    MISTRAL_METADATA_FILE,
    
    # Log files
    BUILD_INDEX_LOG,
    
    # Indexing configuration
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    ADVANCED_EMBEDDING_MODEL,
    DEFAULT_FAISS_TYPE,
    DEFAULT_NLIST,
    DEFAULT_TRAIN_SAMPLE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    
    # Retrieval configuration
    DEFAULT_TOP_K,
    EMBED_PROMPT,
    
    # Generation configuration
    DEFAULT_GEN_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    
    # Environment settings
    OMP_NUM_THREADS,
    MKL_NUM_THREADS,
    USE_GPU,
    DEVICE,
    MPS_AVAILABLE
)

from veritas.utils import (
    # Utility functions
    get_project_root,
    get_model_path,
    get_data_path,
    get_log_path,
    get_script_path,
    ensure_directories,
    resolve_path,
    ensure_parent_dirs
)

from veritas.mps_utils import (
    get_optimal_device,
    optimize_for_mps,
    get_optimal_batch_size,
    prepare_inputs_for_mps,
    get_memory_info,
    log_memory_info
)

__version__ = "0.1.0"
__author__ = "Veritas Team"

__all__ = [
    "RAGSystem",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_FAISS_TYPE",
    "DEFAULT_NLIST",
    "DEFAULT_BATCH_SIZE",
    "FAISS_INDEX_FILE",
    "METADATA_FILE",
    "RAG_CHUNKS_FILE",
    "DEVICE",
    "MPS_AVAILABLE",
    "ensure_parent_dirs",
    "get_optimal_device",
    "optimize_for_mps",
    "get_optimal_batch_size",
    "prepare_inputs_for_mps",
    "get_memory_info",
    "log_memory_info"
] 