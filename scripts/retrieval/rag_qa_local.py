#!/usr/bin/env python3
"""
rag_qa_local.py

A minimal RAG QA loop without any external API:
 1. Dense retrieval via FAISS + SentenceTransformers
 2. Prompt construction with top-K contexts
 3. Answer generation via a local Hugging Face–style model
"""

import os
import argparse
import pickle
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from config import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEN_MODEL,
    DEFAULT_TOP_K,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DEVICE,
    ensure_directories
)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_INDEX   = str(FAISS_INDEX_FILE)
DEFAULT_META    = str(METADATA_FILE)
DEFAULT_EMBED   = DEFAULT_EMBEDDING_MODEL
DEFAULT_GEN     = DEFAULT_GEN_MODEL
DEFAULT_TOP_K   = DEFAULT_TOP_K

# ─── RETRIEVAL ─────────────────────────────────────────────────────────────────
def retrieve(query, index_path, meta_path, embed_model, top_k):
    embedder = SentenceTransformer(embed_model, device=DEVICE)
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metas = pickle.load(f)

    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype="float32")
    scores, idxs = index.search(q_vec, top_k)

    hits = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        hits.append({
            "score": score,
            "text":  m.get("text", "")[:500],
            "title": m.get("title", ""),
            "id":    m.get("id", "")
        })
    return hits

# ─── GENERATION ────────────────────────────────────────────────────────────────
def generate_answer_local(query, contexts, gen_model):
    # Load tokenizer & model once
    tokenizer = AutoTokenizer.from_pretrained(gen_model)
    model     = AutoModelForCausalLM.from_pretrained(
        gen_model,
        device_map="auto",
        torch_dtype="auto"
    )
    # Build prompt
    system = "You are a helpful assistant. Answer using ONLY the provided context."
    context_block = "\n\n---\n\n".join(contexts)
    prompt = (
        f"{system}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    # Set up a text-generation pipeline
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE
    )
    out = gen(prompt, return_full_text=False)[0]["generated_text"]
    # Trim off the prompt
    return out[len(prompt):].strip()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG QA with local generation")
    parser.add_argument("--index",       type=str, default=DEFAULT_INDEX, help="FAISS index path")
    parser.add_argument("--meta",        type=str, default=DEFAULT_META,  help="Metadata pickle path")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED, help="Embedding model name")
    parser.add_argument("--gen-model",   type=str, default=DEFAULT_GEN,   help="Local LLM model name")
    parser.add_argument("--query",       type=str, required=True,         help="Your question")
    parser.add_argument("--top_k",       type=int, default=DEFAULT_TOP_K, help="Number of passages to retrieve")
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # 1) Retrieve
    hits = retrieve(args.query, args.index, args.meta, args.embed_model, args.top_k)
    if not hits:
        print("⚠️ No passages found.")
        return

    snippets = [h["text"] for h in hits]

    # 2) Generate locally
    answer = generate_answer_local(args.query, snippets, args.gen_model)

    # 3) Display
    print("\n📝 Answer:\n")
    print(answer)
    print("\n🔍 Retrieved passages:")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h['score']:.3f}] {h['title']} (id={h['id']})")

if __name__ == "__main__":
    main()
