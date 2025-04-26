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
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from config import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    ADVANCED_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    OMP_NUM_THREADS,
    MKL_NUM_THREADS,
    ensure_directories
)

# 1) Force single‐threaded BLAS/MKL (to mirror your search script)
os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = MKL_NUM_THREADS

# 2) Load your embedding model to discover its output dimension
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer(ADVANCED_EMBEDDING_MODEL, device="cpu")
except Exception:
    # fallback if you didn't have instructor-xl locally
    model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL, device="cpu")

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
faiss.write_index(index, str(FAISS_INDEX_FILE))

# 6) Write out an empty metadata list
with open(METADATA_FILE, "wb") as f:
    pickle.dump([], f)

print("✅ Created:")
print(f"   • {FAISS_INDEX_FILE}")
print(f"   • {METADATA_FILE}")
