#!/usr/bin/env python3
"""
build_faiss_index.py

Enhanced FAISS index builder with:
 - CLI configurability via argparse
 - Optional GPU (MPS) acceleration
 - Choice of Flat or IVF+PQ indices
 - Dynamic batching and overall progress bars
 - Robust logging for errors and key steps
 - Optional multiprocessing for parallel encoding
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import json
import pickle
import multiprocessing

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("build_faiss_index.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ─── UTILITIES ───────────────────────────────────────────────────────────────
def chunked(iterable, size):
    """Yield successive chunks from iterable of given size."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
def load_embedding_model(model_name: str, use_gpu: bool):
    import torch
    device = "mps" if (use_gpu and sys.platform == "darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
    logging.info(f"Loading model '{model_name}' on device: {device}")
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as e:
        logging.warning(f"Failed to load '{model_name}': {e!r}")
        fallback = "all-MiniLM-L6-v2"
        if model_name != fallback:
            logging.info(f"Falling back to '{fallback}'")
            return SentenceTransformer(fallback, device=device)
        raise

# ─── WORKER INIT & EMBED ──────────────────────────────────────────────────────
_model = None

def init_worker(model_name, use_gpu):
    """Initialize model in each worker process."""
    global _model
    _model = load_embedding_model(model_name, use_gpu)

def embed_batch(batch):
    """Encode a batch of texts in a worker process."""
    global _model
    embs = _model.encode(batch, normalize_embeddings=True)
    return np.vstack(embs).astype("float32")

# ─── INDEX FACTORY ─────────────────────────────────────────────────────────────
def build_faiss_index(dim: int, index_type: str, nlist: int):
    if index_type == "flat":
        logging.info("Using IndexFlatIP")
        return faiss.IndexFlatIP(dim)
    # IVF+PQ
    logging.info(f"Using IndexIVFPQ with nlist={nlist}")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, 8, 8)
    return index

# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from JSON chunks.")
    parser.add_argument("--data",       type=Path, default=Path("rag_chunks.json"), help="Path to JSON with text+meta chunks.")
    parser.add_argument("--index",      type=Path, default=Path("veritas_faiss.index"), help="Output FAISS index path.")
    parser.add_argument("--meta",       type=Path, default=Path("veritas_metadata.pkl"), help="Output metadata pickle path.")
    parser.add_argument("--model",      type=str, default="hkunlp/instructor-xl", help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embeddings.")
    parser.add_argument("--use-gpu",    action="store_true", help="Enable MPS GPU encoding if available.")
    parser.add_argument("--faiss-type", choices=["flat","ivf"], default="flat", help="Type of FAISS index to build.")
    parser.add_argument("--nlist",      type=int, default=100, help="Number of IVF cells (for ivf index).")
    parser.add_argument("--train-sample", type=int, default=10000, help="Number of samples for training IVF (ivf only).")
    parser.add_argument("--workers",    type=int, default=1, help="Number of parallel workers for embedding; >1 uses multiprocessing.")
    args = parser.parse_args()

    # Load data
    if not args.data.exists():
        logging.error(f"Data file not found: {args.data}")
        sys.exit(1)
    with open(args.data, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]
    metas = [{**c.get("meta",{}), "id": c.get("id")} for c in chunks]
    total = len(texts)
    logging.info(f"Loaded {total} chunks.")

    # Prepare model(s)
    if args.workers == 1:
        model = load_embedding_model(args.model, args.use_gpu)
        dim = model.get_sentence_embedding_dimension()
    else:
        # spawn one temp model for dimension
        temp_model = load_embedding_model(args.model, args.use_gpu)
        dim = temp_model.get_sentence_embedding_dimension()
    logging.info(f"Embedding dimension: {dim}")

    # Build index
    index = build_faiss_index(dim, args.faiss_type, args.nlist)

    # IVF training if required
    if args.faiss_type == "ivf":
        sample_count = min(args.train_sample, total)
        logging.info(f"Training IVF on {sample_count} samples.")
        sample_texts = texts[:sample_count]
        if args.workers == 1:
            sample_embs = model.encode(sample_texts, normalize_embeddings=True, show_progress_bar=True)
            index.train(np.vstack(sample_embs).astype("float32"))
        else:
            with multiprocessing.Pool(args.workers, initializer=init_worker, initargs=(args.model, args.use_gpu)) as pool:
                batches = list(chunked(sample_texts, args.batch_size))
                embs_list = pool.map(embed_batch, batches)
                train_embs = np.vstack(embs_list)[:sample_count]
                index.train(train_embs)

    # Embedding & indexing
    logging.info("Starting embedding & indexing...")
    if args.workers == 1:
        with tqdm(total=total, desc="Embedding & indexing") as pbar:
            for batch in chunked(texts, args.batch_size):
                try:
                    embs = model.encode(batch, normalize_embeddings=True)
                    index.add(np.vstack(embs).astype("float32"))
                except Exception:
                    logging.exception("Failed to embed batch start=%d", pbar.n)
                pbar.update(len(batch))
    else:
        with multiprocessing.Pool(args.workers, initializer=init_worker, initargs=(args.model, args.use_gpu)) as pool, \
             tqdm(total=total, desc="Embedding & indexing") as pbar:
            for embs in pool.imap(embed_batch, chunked(texts, args.batch_size)):
                index.add(embs)
                pbar.update(len(embs))

    # Final checks and save
    if index.ntotal != total:
        logging.error(f"Index size {index.ntotal} != metadata size {total}")
        sys.exit(1)
    logging.info(f"Saving index to {args.index}")
    faiss.write_index(index, str(args.index))
    logging.info(f"Saving metadata to {args.meta}")
    with open(args.meta, "wb") as f:
        pickle.dump(metas, f)
    logging.info("✅ Build complete.")

if __name__ == "__main__":
    # optionally remove BLAS caps if desired:
    # os.environ.pop("OMP_NUM_THREADS", None)
    # os.environ.pop("MKL_NUM_THREADS", None)
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    main()