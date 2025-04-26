#!/usr/bin/env python3
"""
stub_index.py

Dynamically creates:
 - veritas_faiss.index   (empty FAISS IndexFlatIP with correct dim)
 - veritas_metadata.pkl  (empty metadata list)
in the current directory.
"""

import os
import pickle

# 1) Force single‐threaded BLAS/MKL (to mirror your search script)
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "1")

# 2) Load your embedding model to discover its output dimension
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer("hkunlp/instructor-xl", device="cpu")
except Exception:
    # fallback if you didn’t have instructor-xl locally
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# 3) Get embedding dimension
try:
    dim = model.get_sentence_embedding_dimension()
except AttributeError:
    # fallback probe
    dim = model.encode(["__dummy__"], normalize_embeddings=True).shape[1]

print(f"Embedding dimension detected: {dim}")

# 4) Build an empty FAISS inner-product index with that dimension
import faiss
index = faiss.IndexFlatIP(dim)

# 5) Write out the stub index file
faiss.write_index(index, "veritas_faiss.index")

# 6) Write out an empty metadata list
with open("veritas_metadata.pkl", "wb") as f:
    pickle.dump([], f)

print("✅ Created:")
print("   • veritas_faiss.index")
print("   • veritas_metadata.pkl")
