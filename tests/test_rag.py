#!/usr/bin/env python3
"""
test_rag.py

A simple script to test the RAG system with the newly created index.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.veritas.rag import RAGSystem
from src.veritas.config import Config

def format_chunk_content(chunk):
    """Format chunk content for display."""
    # Try to get the text content
    text = chunk.get('text', '')
    if not text and 'metadata' in chunk:
        # If text is not directly available, try to get it from metadata
        text = chunk['metadata'].get('text', 'No text available')
    
    # Truncate if too long
    if isinstance(text, str):
        return text[:500] + "..." if len(text) > 500 else text
    return str(text)

def main():
    """Test the RAG system with the new index."""
    try:
        logger.info("Initializing RAG system...")
        rag = RAGSystem()
        
        # Load the index
        logger.info("Loading index...")
        index_path = os.path.join(Config.INDICES_DIR, "latest")
        rag.load_index(index_path)
        
        # Test queries
        test_queries = [
            "What is the main topic of the documents?",
            "Can you summarize the key points?",
            "What are the most important findings?",
            "What are the key concepts discussed?",
            "What are the main arguments presented?",
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # Retrieve relevant chunks with a lower similarity threshold
            logger.info("Retrieving relevant chunks...")
            retrieved_chunks = rag.retrieve(query, k=3, min_score=0.3)
            
            # Log retrieved chunks
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks:")
            for i, chunk in enumerate(retrieved_chunks):
                logger.info(f"\nChunk {i+1} (Score: {chunk.get('score', 'N/A')}):")
                logger.info(f"Text: {format_chunk_content(chunk)}")
                
                # Log metadata if available
                if 'metadata' in chunk:
                    # Filter out large text fields from metadata display
                    display_metadata = {k: v for k, v in chunk['metadata'].items() 
                                     if k != 'text' and not isinstance(v, str) or len(str(v)) < 100}
                    if display_metadata:
                        logger.info(f"Metadata: {display_metadata}")
            
        logger.info("\nRAG system test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing RAG system: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 