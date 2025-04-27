#!/usr/bin/env python3
"""
config.py

Centralized configuration for the Veritas project.
This file contains all configuration parameters used across the project.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "default": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
}

# API configurations
API_CONFIG = {
    "timeout": 30,  # seconds
    "max_retries": 3,
    "retry_delay": 1  # seconds
}

# Logging configurations
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}

# Environment variables (to be loaded from .env file if exists)
API_KEY = os.getenv("VERITAS_API_KEY")
ENVIRONMENT = os.getenv("VERITAS_ENV", "development")

# Development-specific settings
DEBUG = ENVIRONMENT == "development"

# Cache settings
CACHE_CONFIG = {
    "enabled": True,
    "max_size": 1000,
    "ttl": 3600  # Time to live in seconds
}

# Security settings
SECURITY_CONFIG = {
    "ssl_verify": True,
    "allowed_hosts": ["*"] if DEBUG else [],
    "cors_origins": ["*"] if DEBUG else []
}

# Feature flags
FEATURES = {
    "enable_caching": True,
    "enable_logging": True,
    "enable_monitoring": True
}

# ─── Project Structure ──────────────────────────────────────────────────────────
# Base directories
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, SCRIPTS_DIR]:
    directory.mkdir(exist_ok=True)

# ─── Data Files ────────────────────────────────────────────────────────────────
# Input data
INPUT_DATA_FILE = DATA_DIR / "1.json"
RAG_CHUNKS_FILE = DATA_DIR / "rag_chunks.json"

# ─── Model Files ───────────────────────────────────────────────────────────────
# FAISS index and metadata
FAISS_INDEX_FILE = MODELS_DIR / "veritas_faiss.index"
METADATA_FILE = MODELS_DIR / "veritas_metadata.pkl"

# Mistral model
MISTRAL_MODEL_DIR = MODELS_DIR / "mistral-7b"
MISTRAL_FAISS_INDEX = MISTRAL_MODEL_DIR / "veritas_faiss.index"
MISTRAL_METADATA_FILE = MISTRAL_MODEL_DIR / "veritas_metadata.pkl"

# ─── Log Files ────────────────────────────────────────────────────────────────
BUILD_INDEX_LOG = LOGS_DIR / "build_faiss_index.log"

# ─── Indexing Configuration ────────────────────────────────────────────────────
# Chunking settings
DEFAULT_CHUNK_SIZE = 512

# Embedding model settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ADVANCED_EMBEDDING_MODEL = "hkunlp/instructor-xl"

# FAISS index settings
DEFAULT_FAISS_TYPE = "flat"  # Options: "flat", "ivf"
DEFAULT_NLIST = 100
DEFAULT_TRAIN_SAMPLE = 10000
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 1

# ─── Retrieval Configuration ────────────────────────────────────────────────────
# Retrieval settings
DEFAULT_TOP_K = 5
EMBED_PROMPT = "Represent the scientific passage for retrieval: {}"

# ─── Generation Configuration ──────────────────────────────────────────────────
# Generation settings
DEFAULT_GEN_MODEL = "facebook/opt-350m"  # Small, publicly available model
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7

# ─── Environment Settings ──────────────────────────────────────────────────────
# Thread settings
OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "1")
MKL_NUM_THREADS = os.getenv("MKL_NUM_THREADS", "1")

# Device settings
USE_GPU = True  # Set to False to force CPU
DEVICE = "mps" if USE_GPU and os.environ.get("USE_MPS", "1") == "1" else "cpu" 