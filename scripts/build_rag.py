#!/usr/bin/env python3
"""
Script to build and test the RAG system.
"""

import json
from pathlib import Path
from veritas.rag import RAGSystem
from veritas.config import INPUT_DATA_FILE, RAG_CHUNKS_FILE
from veritas.utils import ensure_parent_dirs

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Load input documents
    with open(INPUT_DATA_FILE, 'r') as f:
        data = json.load(f)
    
    # Process documents into chunks
    print("Processing documents into chunks...")
    chunks = rag.process_documents([item["text"] for item in data])
    
    # Save chunks for reference
    ensure_parent_dirs(RAG_CHUNKS_FILE)
    with open(RAG_CHUNKS_FILE, 'w') as f:
        json.dump(chunks, f)
    
    # Build index
    print("Building FAISS index...")
    rag.build_index(chunks)
    
    # Test retrieval and generation
    test_queries = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
    ]
    
    print("\nTesting RAG system with sample queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Retrieve relevant chunks
        retrieved_chunks = rag.retrieve(query)
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Generate response
        response = rag.generate_response(query, retrieved_chunks)
        print(f"Generated response: {response}")

if __name__ == "__main__":
    main() 