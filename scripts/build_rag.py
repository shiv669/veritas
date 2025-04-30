#!/usr/bin/env python3
"""
Script to build and test the RAG system.
"""

import json
import os
import argparse
import sys
from pathlib import Path
import re

# Add the project root to Python path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.veritas.rag import RAGSystem
from src.veritas.config import Config

def get_latest_index():
    """Return the most recent index directory under Config.INDICES_DIR."""
    idx_dir = Config.INDICES_DIR
    subdirs = [d for d in os.listdir(idx_dir) if os.path.isdir(os.path.join(idx_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No index directories found in {idx_dir}")
    # Prefer timestamped dirs like YYYYMMDD_HHMMSS
    ts_pattern = r'^\d{8}_\d{6}$'
    ts_dirs = [d for d in subdirs if re.match(ts_pattern, d)]
    if ts_dirs:
        latest = sorted(ts_dirs)[-1]
    else:
        latest = sorted(subdirs)[-1]
    return os.path.join(idx_dir, latest)

def main():
    # Parse optional index path
    parser = argparse.ArgumentParser(description="Load an existing FAISS index and test the RAG system")
    parser.add_argument('--index-path', '-i', type=str,
                        help="Path to FAISS index directory (overrides auto-detection)")
    args = parser.parse_args()

    # Determine which index to load
    index_path = args.index_path or get_latest_index()
    print(f"Loading FAISS index from {index_path}")
    # Initialize RAG system with the existing index
    rag = RAGSystem(index_path=index_path)
    
    # Test queries
    test_queries = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the conclusions?"
    ]
    
    print("\nTesting retrieval:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            # Print the chunk text from the loaded index
            chunk_data = result.get('chunk', {})
            print(chunk_data.get('text', '')[:500])

if __name__ == "__main__":
    main() 