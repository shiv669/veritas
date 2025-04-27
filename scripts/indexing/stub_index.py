#!/usr/bin/env python3
"""
stub_index.py

Create a stub FAISS index with:
 - Empty index with correct dimensionality
 - Empty metadata file
 - CLI support for index, meta, and model
"""
import os
import sys
import pickle
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from veritas.config import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    ADVANCED_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    OMP_NUM_THREADS,
    MKL_NUM_THREADS,
    ensure_directories,
    resolve_path,
    ensure_parent_dirs
)

# ─── ENV VARS: limit BLAS threads to avoid segfaults ─────────────────────────
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MKL_NUM_THREADS)

import faiss
from sentence_transformers import SentenceTransformer

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model with fallback."""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print(f"Falling back to {FALLBACK_EMBEDDING_MODEL}")
        return SentenceTransformer(FALLBACK_EMBEDDING_MODEL)

# ─── STUB INDEX CREATOR ──────────────────────────────────────────────────────
def create_stub_index(index_file: str, meta_file: str, model_name: str) -> None:
    """
    Create a stub FAISS index with empty metadata.
    
    Args:
        index_file: Path to save the FAISS index
        meta_file: Path to save the metadata
        model_name: Name of the SentenceTransformer model
    """
    # Load model to get embedding dimension
    model = load_embedding_model(model_name)
    
    # Get embedding dimension from a sample text
    sample_text = "This is a sample text to determine the embedding dimension."
    sample_embedding = model.encode(sample_text, normalize_embeddings=True)
    embedding_dim = sample_embedding.shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create empty FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Save index
    faiss.write_index(index, index_file)
    print(f"Created empty FAISS index at {index_file}")
    
    # Create empty metadata
    metadata = []
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Created empty metadata file at {meta_file}")

# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Create a stub FAISS index with empty metadata."
    )
    parser.add_argument(
        "--index", type=str, default=str(FAISS_INDEX_FILE),
        help="Path to save the FAISS index."
    )
    parser.add_argument(
        "--meta", type=str, default=str(METADATA_FILE),
        help="Path to save the metadata."
    )
    parser.add_argument(
        "--model", type=str, default=ADVANCED_EMBEDDING_MODEL,
        help="SentenceTransformer model name to use for embedding dimension."
    )
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()
    ensure_parent_dirs(resolve_path(args.index))
    ensure_parent_dirs(resolve_path(args.meta))

    # Create stub index
    create_stub_index(args.index, args.meta, args.model)

if __name__ == "__main__":
    main()
