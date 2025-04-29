#!/usr/bin/env python3
"""
index_chunks_parallel.py

Script to index the processed chunks into the RAG system using multiple CPUs.
This script loads the chunks from data/chunks.json and builds a FAISS index in parallel.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import List, Dict, Any

from veritas.rag import RAGSystem
from veritas.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAISS_TYPE,
    DEFAULT_NLIST,
    DEFAULT_BATCH_SIZE,
    FAISS_INDEX_FILE,
    METADATA_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the chunks file path
CHUNKS_FILE = Path("data/chunks.json")

def process_batch(args):
    """Process a batch of texts to generate embeddings."""
    batch_texts, model_name = args
    model = SentenceTransformer(model_name)
    embeddings = model.encode(batch_texts, normalize_embeddings=True)
    return embeddings

def main():
    """Main function to index chunks into the RAG system using multiple CPUs."""
    # Check if chunks file exists
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        return
    
    # Load chunks
    logger.info(f"Loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize RAG system
    logger.info("Initializing RAG system")
    rag = RAGSystem(
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        faiss_type=DEFAULT_FAISS_TYPE,
        nlist=DEFAULT_NLIST,
        batch_size=DEFAULT_BATCH_SIZE
    )
    
    # Prepare batches for parallel processing
    texts = [chunk["text"] for chunk in chunks]
    num_cpus = multiprocessing.cpu_count()
    batch_size = max(1, len(texts) // (num_cpus * 4))  # Divide work among CPUs
    
    logger.info(f"Using {num_cpus} CPUs for parallel processing")
    logger.info(f"Processing in batches of {batch_size} texts")
    
    # Process batches in parallel
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    batch_args = [(batch, DEFAULT_EMBEDDING_MODEL) for batch in batches]
    
    embeddings_list = []
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(process_batch, args) for args in batch_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings"):
            embeddings_list.append(future.result())
    
    # Combine embeddings
    embeddings = np.vstack(embeddings_list)
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    if DEFAULT_FAISS_TYPE == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif DEFAULT_FAISS_TYPE == "ivf":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, DEFAULT_NLIST, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    
    # Add vectors to index
    logger.info("Adding vectors to index")
    index.add(embeddings)
    
    # Save index and metadata
    logger.info(f"Saving index to {FAISS_INDEX_FILE}")
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    
    logger.info(f"Saving metadata to {METADATA_FILE}")
    with open(METADATA_FILE, 'w') as f:
        json.dump(chunks, f)
    
    logger.info("Index built successfully")
    
    # Test retrieval
    logger.info("Testing retrieval with a sample query")
    test_query = "What are the effects of unionization on workplace safety?"
    results = rag.retrieve(test_query, k=3)
    
    logger.info(f"Sample query: '{test_query}'")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1} (score: {result['score']:.4f}):")
        logger.info(f"  {result['text'][:200]}...")

if __name__ == "__main__":
    main() 