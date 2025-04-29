#!/usr/bin/env python3
"""
config.py

Centralized configuration for the Veritas project.
This file contains all configuration parameters used across the project.
"""

import os
import logging
from pathlib import Path
import torch

# Configure logging
logger = logging.getLogger(__name__)

# ─── Project Structure ──────────────────────────────────────────────────────────
# Base directories
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"
INDEX_DIR = DATA_DIR / "indices"
CHUNKS_DIR = DATA_DIR / "chunks"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, SCRIPTS_DIR, INDEX_DIR, CHUNKS_DIR]:
    directory.mkdir(exist_ok=True)

# ─── Device Configuration ───────────────────────────────────────────────────────
# Device settings
USE_GPU = True  # Set to False to force CPU
USE_MPS = os.environ.get("USE_MPS", "1") == "1"  # Allow environment override

# Check if MPS is available
try:
    MPS_AVAILABLE = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except (ImportError, AttributeError):
    MPS_AVAILABLE = False

def get_device():
    """Get the optimal device for the system."""
    if USE_GPU and USE_MPS and MPS_AVAILABLE:
        return "mps"
    elif USE_GPU and torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
logger.info(f"Using {DEVICE} for acceleration")

# ─── Model Configuration ────────────────────────────────────────────────────────
# Default models
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_GEN_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADVANCED_EMBEDDING_MODEL = "hkunlp/instructor-xl"

# Model settings
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
MAX_RESPONSE_LENGTH = 500

# ─── Performance Settings ────────────────────────────────────────────────────────
# Batch sizes
DEFAULT_BATCH_SIZE = 64
OPTIMIZED_BATCH_SIZE = 32 if DEVICE == "mps" else 64
EMBED_BATCH_SIZE = 32 if DEVICE == "mps" else 64
GEN_BATCH_SIZE = 4 if DEVICE == "mps" else 8

# Length limits
EMBED_MAX_LENGTH = 512
GEN_MAX_LENGTH = 2048

# Thread settings
OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "1")
MKL_NUM_THREADS = os.getenv("MKL_NUM_THREADS", "1")

# ─── Indexing Configuration ────────────────────────────────────────────────────
# Chunking settings
DEFAULT_CHUNK_SIZE = 512

# FAISS index settings
DEFAULT_FAISS_TYPE = "flat"  # Options: "flat", "ivf"
DEFAULT_NLIST = 100  # Number of clusters for IVF index
DEFAULT_TRAIN_SAMPLE = 10000
DEFAULT_WORKERS = 1

# ─── Retrieval Configuration ────────────────────────────────────────────────────
# Retrieval settings
DEFAULT_TOP_K = 5
EMBED_PROMPT = "Represent the scientific passage for retrieval: {}"

# ─── File Paths ────────────────────────────────────────────────────────────────
# Input data
INPUT_DATA_FILE = DATA_DIR / "1.json"
RAG_CHUNKS_FILE = CHUNKS_DIR / "chunks.json"

# FAISS index and metadata
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.pkl"

# Log files
BUILD_INDEX_LOG = LOGS_DIR / "build_faiss_index.log"

# ─── API Configuration ────────────────────────────────────────────────────────
API_CONFIG = {
    "timeout": 30,  # seconds
    "max_retries": 3,
    "retry_delay": 1  # seconds
}

# ─── Logging Configuration ────────────────────────────────────────────────────
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}

# ─── Environment Settings ──────────────────────────────────────────────────────
API_KEY = os.getenv("VERITAS_API_KEY")
ENVIRONMENT = os.getenv("VERITAS_ENV", "development")
DEBUG = ENVIRONMENT == "development"

# ─── Cache Settings ───────────────────────────────────────────────────────────
CACHE_CONFIG = {
    "enabled": True,
    "max_size": 1000,
    "ttl": 3600  # Time to live in seconds
}

# ─── Security Settings ────────────────────────────────────────────────────────
SECURITY_CONFIG = {
    "ssl_verify": True,
    "allowed_hosts": ["*"] if DEBUG else [],
    "cors_origins": ["*"] if DEBUG else []
}

# ─── Feature Flags ───────────────────────────────────────────────────────────
FEATURES = {
    "enable_caching": True,
    "enable_logging": True,
    "enable_monitoring": True
}

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Model directories
MODELS_DIR = ROOT_DIR / "models"
MISTRAL_MODEL_DIR = MODELS_DIR / "mistral"
EMBEDDING_MODEL_DIR = MODELS_DIR / "embeddings"
FAISS_INDEX_DIR = MODELS_DIR / "faiss"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, MISTRAL_MODEL_DIR, EMBEDDING_MODEL_DIR, FAISS_INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
}

# Model parameters
TEMPERATURE = MODEL_CONFIG["temperature"]
TOP_P = MODEL_CONFIG["top_p"]
REPETITION_PENALTY = MODEL_CONFIG["repetition_penalty"]
MAX_SEQ_LENGTH = MODEL_CONFIG["max_tokens"]

# Device settings
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Performance settings
EMBED_BATCH_SIZE = 32 if DEVICE == "mps" else 64
GEN_BATCH_SIZE = 4 if DEVICE == "mps" else 8
NUM_BEAMS = 1

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# Cache settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Data directories
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Create data directories
for dir_path in [DATA_DIR, INPUT_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"
MISTRAL_FAISS_INDEX = FAISS_INDEX_DIR / "mistral_index.faiss"
MISTRAL_METADATA_FILE = FAISS_INDEX_DIR / "mistral_metadata.json"
EMBEDDING_FAISS_INDEX = FAISS_INDEX_DIR / "embedding_index.faiss"
EMBEDDING_METADATA_FILE = FAISS_INDEX_DIR / "embedding_metadata.json"

# API settings
API_TIMEOUT = 30
API_RETRIES = 3
API_RETRY_DELAY = 1

# Environment variables
API_KEY_ENV = "VERITAS_API_KEY"
ENV = os.getenv("VERITAS_ENV", "development")

# Security settings
SSL_VERIFY = True
ALLOWED_HOSTS = ["*"]

# Feature flags
FEATURE_FLAGS = {
    "caching": True,
    "logging": True,
    "monitoring": True,
}