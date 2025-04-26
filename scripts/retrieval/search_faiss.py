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

# ─── ENV VARS: limit BLAS threads to avoid segfaults ─────────────────────────
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "1")

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
def load_model(model_name: str):
    """Load specified model, fallback to MiniLM if it fails."""
    try:
        print(f"🔄 Loading model '{model_name}' on CPU…")
        return SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        fallback = "all-MiniLM-L6-v2"
        print(f"⚠️ Failed to load '{model_name}': {e!r}. Falling back to '{fallback}'…")
        return SentenceTransformer(fallback, device="cpu")

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
        "--index", type=str, default="veritas_faiss.index",
        help="Path to the FAISS index file."
    )
    parser.add_argument(
        "--meta", type=str, default="veritas_metadata.pkl",
        help="Path to the pickled metadata file."
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name to use for query embeddings."
    )
    parser.add_argument(
        "--query", type=str, required=True,
        help="The search query string."
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top results to return."
    )
    args = parser.parse_args()

    # Load components
    model = load_model(args.model)
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
