#!/usr/bin/env python3
"""
Script to build and test the RAG system.
"""

import json
from pathlib import Path
from veritas.rag import RAGSystem

def main():
    # Load processed documents
    with open("data/processed_1.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        documents = [doc["content"] for doc in data["documents"]]
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Process documents into chunks
    print("Processing documents into chunks...")
    chunks = rag.process_documents(documents)
    
    # Build FAISS index
    print("Building FAISS index...")
    rag.build_index(chunks)
    
    # Test queries
    test_queries = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the conclusions?"
    ]
    
    print("\nTesting retrieval:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(result["text"])

if __name__ == "__main__":
    main() 