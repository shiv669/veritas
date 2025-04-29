#!/usr/bin/env python3
"""
test_rag_workflow.py

A script to test the complete RAG workflow:
1. Process documents into chunks
2. Build FAISS index
3. Test retrieval with queries
"""

import json
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from veritas.rag import RAGSystem
from veritas.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAISS_TYPE,
    DEFAULT_NLIST,
    DEFAULT_BATCH_SIZE,
    FAISS_INDEX_FILE,
    METADATA_FILE
)

def create_sample_documents():
    """Create sample documents for testing."""
    return [
        {
            "text": """
            The RAG (Retrieval-Augmented Generation) system is a powerful approach to question answering.
            It combines the strengths of large language models with efficient document retrieval.
            The system first retrieves relevant documents, then uses them as context for generating answers.
            This approach helps reduce hallucinations and improves answer accuracy.
            """,
            "metadata": {
                "source": "rag_overview",
                "title": "Introduction to RAG Systems",
                "author": "AI Research Team"
            }
        },
        {
            "text": """
            FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.
            It's particularly useful for building search indices for large-scale document retrieval.
            FAISS supports various index types, from simple flat indices to more complex quantized indices.
            The library is optimized for both CPU and GPU architectures.
            """,
            "metadata": {
                "source": "faiss_overview",
                "title": "Understanding FAISS",
                "author": "ML Engineering Team"
            }
        },
        {
            "text": """
            Sentence Transformers are neural networks that convert sentences into dense vector representations.
            These embeddings capture semantic meaning and can be used for various NLP tasks.
            The models are pre-trained on large text corpora and can be fine-tuned for specific applications.
            They're particularly useful for building semantic search systems.
            """,
            "metadata": {
                "source": "embeddings_overview",
                "title": "Sentence Transformers Explained",
                "author": "NLP Research Team"
            }
        }
    ]

def main():
    """Test the complete RAG workflow."""
    try:
        # Create sample documents
        logger.info("Creating sample documents...")
        documents = create_sample_documents()
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag = RAGSystem(
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            faiss_type=DEFAULT_FAISS_TYPE,
            nlist=DEFAULT_NLIST,
            batch_size=DEFAULT_BATCH_SIZE
        )
        
        # Process documents into chunks
        logger.info("Processing documents into chunks...")
        chunks = []
        for doc in documents:
            doc_chunks = rag.process_documents([doc["text"]])
            # Add metadata to each chunk
            for chunk in doc_chunks:
                chunk["metadata"] = doc["metadata"]
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        rag.build_index(chunks)
        
        logger.info(f"Index built successfully and saved to {FAISS_INDEX_FILE}")
        logger.info(f"Metadata saved to {METADATA_FILE}")
        
        # Test retrieval with various queries
        test_queries = [
            "What is RAG and how does it work?",
            "Explain FAISS and its purpose",
            "How do sentence transformers help in search?",
            "What are the benefits of using RAG for question answering?"
        ]
        
        logger.info("\nTesting retrieval with sample queries:")
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            results = rag.retrieve(query, k=2, min_score=0.3)
            
            for i, result in enumerate(results, 1):
                logger.info(f"\nResult {i} (Score: {result.get('score', 'N/A')}):")
                logger.info(f"Text: {result['text'][:200]}...")
                if 'metadata' in result:
                    logger.info(f"Source: {result['metadata'].get('source', 'Unknown')}")
                    logger.info(f"Title: {result['metadata'].get('title', 'Unknown')}")
        
        logger.info("\nRAG workflow test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in RAG workflow test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 