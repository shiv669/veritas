#!/usr/bin/env python3
"""
rag_qa_local.py

RAG-based question answering with:
 - Local Mistral model
 - FAISS-based retrieval
 - CLI support for index, meta, model, and query
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
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    EMBED_PROMPT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    ensure_directories,
    resolve_path,
    ensure_parent_dirs
)

import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# ─── LOAD INDEX & METADATA ────────────────────────────────────────────────────
def load_index_and_meta(index_path: str, meta_path: str):
    """Load FAISS index and metadata."""
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
    """Query the RAG system."""
    prompt = EMBED_PROMPT.format(question)
    q_emb = model.encode([prompt], normalize_embeddings=True)
    q_emb = q_emb.astype("float32")
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
            "text":  m.get("text", "")
        })
    return results

# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAG-based question answering with local Mistral model."
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
        "--query", type=str, required=True,
        help="The question to answer."
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
    model = load_embedding_model(args.model)
    index, metas = load_index_and_meta(args.index, args.meta)

    # Early exit if empty
    if index.ntotal == 0 or not metas:
        print("⚠️ Index or metadata is empty. Run `build_faiss_index.py` to populate your index.")
        sys.exit(0)

    # Run query
    results = query_rag(model, index, metas, args.query, args.top_k)
    if not results:
        print("No results found.")
        return
    for r in results:
        print(f"[{r['score']:.4f}] {r['id']} — {r['title']} ({r['year']})")

if __name__ == "__main__":
    main()
