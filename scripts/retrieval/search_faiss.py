#!/usr/bin/env python3
"""
search_faiss.py

FAISS RAG search script with:
 - Single-threaded BLAS/MKL to prevent segfaults
 - Optional PyTorch thread limit
 - Encapsulated main() to avoid unintended parallel init
 - CLI support for index, meta, model, query, and top-k
 - Early exit if index or metadata is empty
"""
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from veritas import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    OMP_NUM_THREADS,
    MKL_NUM_THREADS,
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
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
def load_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model with fallback."""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print(f"Falling back to {FALLBACK_EMBEDDING_MODEL}")
        return SentenceTransformer(FALLBACK_EMBEDDING_MODEL)

# ─── LOAD INDEX & METADATA ────────────────────────────────────────────────────
def load_index_and_meta(index_path: str, meta_path: str):
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        sys.exit(1)
    if not os.path.exists(meta_path):
        print(f"Metadata file not found: {meta_path}")
        sys.exit(1)
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metas = pickle.load(f)
    return index, metas

# ─── QUERY FUNCTION ───────────────────────────────────────────────────────────
def query_rag(model, index, metas, question: str, top_k: int=5):
    prompt = f"Represent the scientific passage for retrieval: {question}"
    q_emb = model.encode([prompt], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        results.append({
            "score": float(score),
            "id":    m.get("id"),
            "title": m.get("title"),
            "authors": m.get("authors", []),
            "year":  m.get("year"),
            "source": m.get("source", ""),
        })
    return results

# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Search a FAISS index with SentenceTransformers embeddings."
    )
    parser.add_argument(
        "--index", type=str, default=str(FAISS_INDEX_FILE),
        help="Path to the FAISS index file."
    )
    parser.add_argument(
        "--meta", type=str, default=str(METADATA_FILE),
        help="Path to the pickled metadata file."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name to use for query embeddings."
    )
    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help="Number of top results to return."
    )
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()
    ensure_parent_dirs(resolve_path(args.index))
    ensure_parent_dirs(resolve_path(args.meta))

    # Load components
    model = load_model(args.model)
    index, metas = load_index_and_meta(args.index, args.meta)

    # Early exit if empty
    if index.ntotal == 0 or not metas:
        print("⚠️ Index or metadata is empty. Run `build_faiss_index.py` to populate your index.")
        sys.exit(0)

    # Interactive search loop
    while True:
        query = input("\nEnter your search query (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        # Encode the query
        query_vector = model.encode([query])[0].astype('float32')

        # Search the index
        distances, indices = index.search(query_vector.reshape(1, -1), args.top_k)

        # Print results
        print("\nSearch results:")
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(metas):
                result = metas[idx]
                print(f"\n{i+1}. Score: {1 - dist:.4f}")
                print(f"Text: {result.get('text', '')[:200]}...")
                print(f"Source: {result.get('source', 'Unknown')}")
            else:
                print(f"\n{i+1}. Invalid index: {idx}")

if __name__ == "__main__":
    main()
