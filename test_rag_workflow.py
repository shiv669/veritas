#!/usr/bin/env python3
"""
Comprehensive test script for the Veritas RAG system.
Tests the entire workflow from document chunking to retrieval and response generation.
"""

import logging
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from veritas import RAGSystem
from veritas.config import (
    # Model settings
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEN_MODEL,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    
    # Device settings
    DEVICE,
    get_device,
    
    # Performance settings
    EMBED_BATCH_SIZE,
    GEN_BATCH_SIZE,
    MAX_SEQ_LENGTH as EMBED_MAX_LENGTH,
    MAX_SEQ_LENGTH as GEN_MAX_LENGTH,
    
    # Indexing settings
    CHUNK_SIZE as DEFAULT_CHUNK_SIZE
)
from veritas.mps_utils import get_memory_info, log_memory_info
from veritas.chunking import Chunker, ChunkingConfig, ChunkingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGWorkflowTest:
    """Test suite for the complete RAG workflow."""
    
    def __init__(self, device: str = None):
        """Initialize the test suite."""
        self.device = device or get_device()
        self.rag = None
        self.test_documents = []
        self.chunks = []
        self.embeddings = None
        self.results = {}
        
        # Load test data
        self._load_test_data()
    
    def _load_test_data(self):
        """Load test documents for the workflow test."""
        # Load sample document
        doc_path = Path("test_docs/sample.txt")
        if not doc_path.exists():
            raise FileNotFoundError(f"Test document not found at {doc_path}")
        
        with open(doc_path, "r") as f:
            content = f.read()
        
        self.test_documents = [{
            "content": content,
            "metadata": {
                "source": "test_docs/sample.txt",
                "type": "documentation",
                "topic": "RAG systems",
                "created_at": "2024-03-21"
            }
        }]
        
        # Load additional test documents if available
        additional_docs_dir = Path("test_docs/additional")
        if additional_docs_dir.exists():
            for doc_file in additional_docs_dir.glob("*.txt"):
                with open(doc_file, "r") as f:
                    content = f.read()
                
                self.test_documents.append({
                    "content": content,
                    "metadata": {
                        "source": str(doc_file),
                        "type": "documentation",
                        "topic": "Additional test data",
                        "created_at": "2024-03-21"
                    }
                })
        
        logger.info(f"Loaded {len(self.test_documents)} test documents")
    
    def test_chunking(self):
        """Test document chunking."""
        logger.info("\n=== Testing Document Chunking ===")
        
        # Initialize chunker with different strategies
        chunking_strategies = [
            ChunkingStrategy.FIXED,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.PARAGRAPH,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.HYBRID
        ]
        
        self.results["chunking"] = {}
        
        for strategy in chunking_strategies:
            logger.info(f"\nTesting {strategy.value} chunking strategy...")
            
            # Configure chunker
            config = ChunkingConfig(
                strategy=strategy,
                chunk_size=1024,  # Increased from 256
                overlap=100,  # Increased from 25
                min_chunk_size=25,  # Decreased from 50
                max_chunk_size=2048  # Increased from 500
            )
            chunker = Chunker(config)
            
            # Process each document
            all_chunks = []
            for doc_idx, doc in enumerate(self.test_documents):
                start_time = time.time()
                # Extract content from document dictionary
                content = doc["content"] if isinstance(doc, dict) else doc
                chunks = chunker.chunk_text(content)
                duration = time.time() - start_time
                
                # Add document index to chunk metadata
                for chunk in chunks:
                    chunk["doc_id"] = doc_idx
                
                all_chunks.extend(chunks)
                
                logger.info(f"Document {doc_idx}: Generated {len(chunks)} chunks in {duration:.2f} seconds")
            
            # Store results
            self.results["chunking"][strategy.value] = {
                "num_chunks": len(all_chunks),
                "avg_chunk_size": np.mean([len(chunk["text"].split()) for chunk in all_chunks]),
                "chunks": all_chunks
            }
            
            logger.info(f"Total chunks: {len(all_chunks)}")
            logger.info(f"Average chunk size: {self.results['chunking'][strategy.value]['avg_chunk_size']:.1f} words")
        
        # Use the hybrid strategy for further testing
        self.chunks = self.results["chunking"][ChunkingStrategy.HYBRID.value]["chunks"]
        logger.info(f"Selected {len(self.chunks)} chunks from hybrid strategy for further testing")
    
    def test_embedding_generation(self):
        """Test embedding generation."""
        logger.info("\n=== Testing Embedding Generation ===")
        
        # Initialize RAG system
        self.rag = RAGSystem(device=self.device)
        
        # Generate embeddings for chunks
        texts = [chunk["text"] for chunk in self.chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        start_time = time.time()
        
        self.embeddings = self.rag.generate_embeddings(texts)
        
        duration = time.time() - start_time
        logger.info(f"Generated {len(texts)} embeddings in {duration:.2f} seconds")
        logger.info(f"Embedding dimension: {self.embeddings.shape[1]}")
        
        # Store results
        self.results["embedding"] = {
            "num_embeddings": len(texts),
            "embedding_dim": self.embeddings.shape[1],
            "duration": duration,
            "throughput": len(texts) / duration
        }
    
    def test_index_building(self):
        """Test FAISS index building."""
        logger.info("\n=== Testing Index Building ===")
        
        if self.embeddings is None:
            logger.error("Embeddings not generated. Run test_embedding_generation first.")
            return
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        start_time = time.time()
        
        self.rag.process_documents(self.test_documents)
        
        duration = time.time() - start_time
        logger.info(f"Built FAISS index in {duration:.2f} seconds")
        
        # Store results
        self.results["index"] = {
            "num_vectors": self.rag.index.ntotal,
            "dimension": self.rag.index.d,
            "duration": duration
        }
    
    def test_retrieval(self):
        """Test retrieval functionality."""
        logger.info("\n=== Testing Retrieval ===")
        
        if self.rag.index is None:
            logger.error("Index not built. Run test_index_building first.")
            return
        
        # Test queries
        test_queries = [
            "What are the key components of a RAG system?",
            "Why is chunking important in RAG and what are the best practices?",
            "What are the main benefits of using RAG for question answering?",
            "What implementation considerations should be taken into account when building a RAG system?"
        ]
        
        self.results["retrieval"] = {}
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # Test different k values
            for k in [1, 3, 5, 10]:
                start_time = time.time()
                results = self.rag.retrieve(query, k=k)
                duration = time.time() - start_time
                
                logger.info(f"k={k}: Retrieved {len(results)} results in {duration:.2f} seconds")
                
                # Store results
                if query not in self.results["retrieval"]:
                    self.results["retrieval"][query] = {}
                
                self.results["retrieval"][query][f"k={k}"] = {
                    "num_results": len(results),
                    "duration": duration,
                    "scores": [r["score"] for r in results] if results else []
                }
                
                # Log retrieved chunks
                if results:
                    for i, result in enumerate(results):
                        logger.info(f"Result {i+1} (score: {result['score']:.3f}):")
                        logger.info(f"Text: {result['text'][:200]}...")
    
    def test_response_generation(self):
        """Test response generation."""
        logger.info("\n=== Testing Response Generation ===")
        
        if self.rag.index is None:
            logger.error("Index not built. Run test_index_building first.")
            return
        
        # Test queries
        test_queries = [
            "What are the key components of a RAG system?",
            "Why is chunking important in RAG and what are the best practices?",
            "What are the main benefits of using RAG for question answering?",
            "What implementation considerations should be taken into account when building a RAG system?"
        ]
        
        self.results["generation"] = {}
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # Retrieve context
            context = self.rag.retrieve(query, k=5)
            context_text = "\n\n".join([chunk["text"] for chunk in context])
            
            # Generate response
            start_time = time.time()
            response = self.rag.generate_response(query, context_text)
            duration = time.time() - start_time
            
            logger.info(f"Generated response in {duration:.2f} seconds")
            logger.info(f"Response: {response}")
            
            # Store results
            self.results["generation"][query] = {
                "duration": duration,
                "response_length": len(response.split()),
                "response": response
            }
    
    def run_workflow_test(self):
        """Run the complete workflow test."""
        logger.info(f"Starting RAG workflow test on device: {self.device}")
        log_memory_info()
        
        # Run each test in sequence
        self.test_chunking()
        self.test_embedding_generation()
        self.test_index_building()
        self.test_retrieval()
        self.test_response_generation()
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save test results to a JSON file."""
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"rag_workflow_test_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            if key == "generation":
                # Only save response lengths for generation results
                serializable_results[key] = {
                    query: {
                        "duration": data["duration"],
                        "response_length": data["response_length"]
                    }
                    for query, data in value.items()
                }
            else:
                serializable_results[key] = value
        
        results_data = {
            "device": self.device,
            "model_settings": {
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "generation_model": DEFAULT_GEN_MODEL,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "repetition_penalty": REPETITION_PENALTY
            },
            "performance_settings": {
                "embed_batch_size": EMBED_BATCH_SIZE,
                "gen_batch_size": GEN_BATCH_SIZE,
                "embed_max_length": EMBED_MAX_LENGTH,
                "gen_max_length": GEN_MAX_LENGTH,
                "chunk_size": DEFAULT_CHUNK_SIZE
            },
            "results": serializable_results
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"\nTest results saved to: {results_file}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print a summary of test results."""
        logger.info("\n=== RAG Workflow Test Summary ===")
        logger.info(f"Device: {self.device}")
        logger.info(f"Embedding Model: {DEFAULT_EMBEDDING_MODEL}")
        logger.info(f"Generation Model: {DEFAULT_GEN_MODEL}")
        
        # Chunking summary
        if "chunking" in self.results:
            logger.info("\nChunking Results:")
            for strategy, data in self.results["chunking"].items():
                logger.info(f"  {strategy}: {data['num_chunks']} chunks, avg size: {data['avg_chunk_size']:.1f} words")
        
        # Embedding summary
        if "embedding" in self.results:
            logger.info("\nEmbedding Results:")
            logger.info(f"  Generated {self.results['embedding']['num_embeddings']} embeddings")
            logger.info(f"  Embedding dimension: {self.results['embedding']['embedding_dim']}")
            logger.info(f"  Throughput: {self.results['embedding']['throughput']:.2f} embeddings/second")
        
        # Index summary
        if "index" in self.results:
            logger.info("\nIndex Results:")
            logger.info(f"  Built index with {self.results['index']['num_vectors']} vectors")
            logger.info(f"  Index dimension: {self.results['index']['dimension']}")
        
        # Retrieval summary
        if "retrieval" in self.results:
            logger.info("\nRetrieval Results:")
            for query, k_results in self.results["retrieval"].items():
                logger.info(f"  Query: {query[:50]}...")
                for k, data in k_results.items():
                    logger.info(f"    {k}: {data['num_results']} results in {data['duration']:.2f} seconds")
        
        # Generation summary
        if "generation" in self.results:
            logger.info("\nGeneration Results:")
            for query, data in self.results["generation"].items():
                logger.info(f"  Query: {query[:50]}...")
                logger.info(f"    Generated {data['response_length']} words in {data['duration']:.2f} seconds")

def main():
    try:
        workflow_test = RAGWorkflowTest()
        workflow_test.run_workflow_test()
    except Exception as e:
        logger.error(f"Error during workflow test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 