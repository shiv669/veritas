#!/usr/bin/env python3
"""
index_chunks.py

Script to index the processed chunks into the RAG system.
This script loads the documents from data/processed_1.json and builds a FAISS index.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm

from veritas.rag import RAGSystem
from veritas.config import (
    FAISS_INDEX_FILE,
    METADATA_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the input file path
INPUT_FILE = Path("data/processed_1.json")

def main():
    """Main function to index documents into the RAG system."""
    # Check if input file exists
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return
    
    # Load documents
    logger.info(f"Loading documents from {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict) and "documents" in data:
            documents = data["documents"]
        else:
            documents = data
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Initialize RAG system
    logger.info("Initializing RAG system")
    logger.info("Using CPU for embedding generation")
    rag = RAGSystem(device="cpu")
    
    # Process documents and build index
    logger.info("Processing documents and building FAISS index")
    rag.process_documents(documents)
    
    logger.info(f"Index built successfully and saved to {FAISS_INDEX_FILE}")
    logger.info(f"Metadata saved to {METADATA_FILE}")
    
    # Test retrieval
    logger.info("Testing retrieval with a sample query")
    test_query = "What are the key components of a RAG system?"
    results = rag.retrieve(test_query, k=3)
    
    logger.info(f"\nSample query: '{test_query}'")
    for i, result in enumerate(results, 1):
        logger.info(f"\nResult {i} (Score: {result['score']:.4f}):")
        logger.info(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])

if __name__ == "__main__":
    main() 