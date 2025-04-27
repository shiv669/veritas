#!/usr/bin/env python3
"""
build_faiss_index.py

Build a FAISS index from RAG chunks with:
 - Single-threaded BLAS/MKL to prevent segfaults
 - Optional PyTorch thread limit
 - Encapsulated main() to avoid unintended parallel init
 - CLI support for data, index, meta, model, and FAISS parameters
"""
import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from veritas.config import (
    RAG_CHUNKS_FILE,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    BUILD_INDEX_LOG,
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    ADVANCED_EMBEDDING_MODEL,
    DEFAULT_FAISS_TYPE,
    DEFAULT_NLIST,
    DEFAULT_TRAIN_SAMPLE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    OMP_NUM_THREADS,
    MKL_NUM_THREADS,
    USE_GPU,
    DEVICE,
    ensure_directories,
    resolve_path,
    ensure_parent_dirs
)

# ─── ENV VARS: limit BLAS threads to avoid segfaults ─────────────────────────
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MKL_NUM_THREADS)

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BUILD_INDEX_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model with fallback."""
    try:
        logger.info(f"Loading model '{model_name}' on {DEVICE}...")
        return SentenceTransformer(model_name, device=DEVICE)
    except Exception as e:
        fallback = FALLBACK_EMBEDDING_MODEL
        logger.warning(f"Failed to load '{model_name}': {e!r}. Falling back to '{fallback}'...")
        return SentenceTransformer(fallback, device=DEVICE)

# ─── FAISS INDEX BUILDER ──────────────────────────────────────────────────────
def build_faiss_index(
    data_file: str,
    index_file: str,
    meta_file: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    faiss_type: str = DEFAULT_FAISS_TYPE,
    nlist: int = DEFAULT_NLIST,
    train_sample: int = DEFAULT_TRAIN_SAMPLE,
    workers: int = DEFAULT_WORKERS
) -> None:
    """
    Build a FAISS index from RAG chunks.
    
    Args:
        data_file: Path to the RAG chunks JSON file
        index_file: Path to save the FAISS index
        meta_file: Path to save the metadata
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for encoding
        faiss_type: Type of FAISS index (flat, ivf)
        nlist: Number of clusters for IVF index
        train_sample: Number of samples to use for training
        workers: Number of workers for encoding
    """
    # Load data
    logger.info(f"Loading data from {data_file}...")
    with open(data_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    if not chunks:
        logger.error("No chunks found in the data file.")
        return
    
    logger.info(f"Loaded {len(chunks)} chunks.")
    
    # Load model
    model = load_embedding_model(model_name)
    
    # Detect embedding dimension
    sample_text = chunks[0]["text"]
    sample_embedding = model.encode(sample_text, normalize_embeddings=True)
    embedding_dim = sample_embedding.shape[0]
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Create FAISS index
    if faiss_type == "flat":
        logger.info("Creating FLAT index...")
        index = faiss.IndexFlatL2(embedding_dim)
    elif faiss_type == "ivf":
        logger.info(f"Creating IVF index with {nlist} clusters...")
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
        
        # Train the index
        logger.info(f"Training index with {min(train_sample, len(chunks))} samples...")
        train_texts = [chunk["text"] for chunk in chunks[:train_sample]]
        train_embeddings = model.encode(train_texts, normalize_embeddings=True)
        index.train(train_embeddings)
    else:
        logger.error(f"Unknown FAISS type: {faiss_type}")
        return
    
    # Encode chunks in batches
    logger.info(f"Encoding {len(chunks)} chunks in batches of {batch_size}...")
    metadata = []
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        batch_texts = [chunk["text"] for chunk in batch]
        
        # Encode batch
        batch_embeddings = model.encode(
            batch_texts, 
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            num_workers=workers
        )
        
        # Add to index
        if faiss_type == "flat":
            index.add(batch_embeddings)
        else:  # ivf
            index.add_with_ids(batch_embeddings, np.arange(i, i+len(batch), dtype=np.int64))
        
        # Collect metadata
        for chunk in batch:
            metadata.append({
                "id": chunk.get("id", ""),
                "title": chunk.get("title", ""),
                "authors": chunk.get("authors", []),
                "year": chunk.get("year", ""),
                "source": chunk.get("source", ""),
                "text": chunk.get("text", "")
            })
    
    # Save index and metadata
    logger.info(f"Saving index to {index_file}...")
    faiss.write_index(index, index_file)
    
    logger.info(f"Saving metadata to {meta_file}...")
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Index built successfully with {index.ntotal} vectors.")

# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from RAG chunks."
    )
    parser.add_argument(
        "--data", type=str, default=str(RAG_CHUNKS_FILE),
        help="Path to the RAG chunks JSON file."
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
        "--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name to use for embeddings."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size for encoding."
    )
    parser.add_argument(
        "--faiss-type", type=str, default=DEFAULT_FAISS_TYPE,
        choices=["flat", "ivf"],
        help="Type of FAISS index to build."
    )
    parser.add_argument(
        "--nlist", type=int, default=DEFAULT_NLIST,
        help="Number of clusters for IVF index."
    )
    parser.add_argument(
        "--train-sample", type=int, default=DEFAULT_TRAIN_SAMPLE,
        help="Number of samples to use for training."
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Number of workers for encoding."
    )
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()
    ensure_parent_dirs(resolve_path(args.index))
    ensure_parent_dirs(resolve_path(args.meta))

    # Build index
    build_faiss_index(
        data_file=args.data,
        index_file=args.index,
        meta_file=args.meta,
        model_name=args.model,
        batch_size=args.batch_size,
        faiss_type=args.faiss_type,
        nlist=args.nlist,
        train_sample=args.train_sample,
        workers=args.workers
    )

if __name__ == "__main__":
    main()