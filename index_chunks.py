#!/usr/bin/env python3
"""
index_chunks.py

Script to index the processed chunks into the RAG system.
This script loads the chunks from data/chunks.json and builds a FAISS index.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm

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

def main():
    """Main function to index chunks into the RAG system."""
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
    
    # Build index
    logger.info("Building FAISS index")
    rag.build_index(chunks)
    
    logger.info(f"Index built successfully and saved to {FAISS_INDEX_FILE}")
    logger.info(f"Metadata saved to {METADATA_FILE}")
    
    # Test retrieval
    logger.info("Testing retrieval with a sample query")
    test_query = "What are the effects of unionization on workplace safety?"
    results = rag.retrieve(test_query, k=3)
    
    logger.info(f"Sample query: '{test_query}'")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1} (score: {result['score']:.4f}):")
        logger.info(f"  {result['text'][:200]}...")
        logger.info(f"  Source: {result.get('source', 'Unknown')}")

if __name__ == "__main__":
    main() 