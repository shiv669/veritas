#!/usr/bin/env python3
"""
Test script for RAG response generation using a sample document.
Tests the RAG system's ability to process and answer queries about RAG concepts.
Includes performance benchmarking for different operations.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from veritas import RAGSystem
from veritas.config import (
    # Model settings
    DEFAULT_EMB_MODEL,
    DEFAULT_GEN_MODEL,
    TEMPERATURE,
    TOP_P,
    
    # Device settings
    DEVICE,
    get_device,
    
    # Performance settings
    EMBED_BATCH_SIZE,
    GEN_BATCH_SIZE
)
from veritas.mps_utils import get_memory_info, log_memory_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test queries focused on RAG concepts
TEST_QUERIES = [
    "What are the key components of a RAG system?",
    "Why is chunking important in RAG and what are the best practices?",
    "What are the main benefits of using RAG for question answering?",
    "What implementation considerations should be taken into account when building a RAG system?"
]

def load_test_document() -> List[Dict]:
    """Load the sample document with metadata."""
    doc_path = Path("test_docs/sample.txt")
    if not doc_path.exists():
        raise FileNotFoundError(f"Test document not found at {doc_path}")
    
    with open(doc_path, "r") as f:
        content = f.read()
    
    # Create document with metadata
    return [{
        "content": content,
        "metadata": {
            "source": "test_docs/sample.txt",
            "type": "documentation",
            "topic": "RAG systems",
            "created_at": "2024-03-21"
        }
    }]

def benchmark_operation(func, *args, **kwargs) -> Tuple[float, Any]:
    """Benchmark a function's execution time and return results."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return duration, result

def main():
    # Get device and log memory info
    device = get_device()
    logger.info(f"Using device: {device}")
    log_memory_info()
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    init_time, rag = benchmark_operation(RAGSystem, device=device)
    logger.info(f"RAG system initialized in {init_time:.2f} seconds")
    
    # Load and process test document
    logger.info("\nLoading and processing test document...")
    documents = load_test_document()
    process_time, _ = benchmark_operation(rag.process_documents, documents)
    logger.info(f"Document processed in {process_time:.2f} seconds")
    
    # Process test queries with benchmarking
    logger.info("\nProcessing test queries...")
    for query in TEST_QUERIES:
        logger.info(f"\nQuery: {query}")
        
        # Benchmark retrieval
        retrieve_time, context = benchmark_operation(rag.retrieve, query)
        logger.info(f"Retrieval completed in {retrieve_time:.2f} seconds")
        
        # Benchmark response generation
        gen_time, response = benchmark_operation(rag.generate_response, query, context)
        logger.info(f"Response generated in {gen_time:.2f} seconds")
        
        logger.info(f"Response: {response}")
        logger.info("-" * 80)
        
        # Log memory usage after each query
        log_memory_info()
    
    # Log final performance summary
    logger.info("\nPerformance Summary:")
    logger.info(f"Device: {device}")
    logger.info(f"Embedding Batch Size: {EMBED_BATCH_SIZE}")
    logger.info(f"Generation Batch Size: {GEN_BATCH_SIZE}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Top-p: {TOP_P}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1) 