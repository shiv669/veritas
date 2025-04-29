#!/usr/bin/env python3
"""
test_rag_system.py

Comprehensive tests for the RAG system, including indexing and retrieval functionality.
"""

import unittest
import tempfile
import json
import shutil
import time
import os
import sys
import gc
import psutil
import traceback
from pathlib import Path
import numpy as np
import faiss
import logging
import pickle
from sentence_transformers import SentenceTransformer
import torch
import random
import string

from veritas.rag import RAGSystem
from index_chunks_parallel import process_group, process_batch, clear_memory, get_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.model_name = "all-MiniLM-L6-v2"
        cls.test_chunks = [
            {
                "text": "The quick brown fox jumps over the lazy dog.",
                "metadata": {"source": "test1", "chunk_index": 0}
            },
            {
                "text": "A quick brown dog jumps over the lazy fox.",
                "metadata": {"source": "test1", "chunk_index": 1}
            },
            {
                "text": "The five boxing wizards jump quickly.",
                "metadata": {"source": "test2", "chunk_index": 0}
            }
        ]
        
        # Save test chunks to a temporary file
        cls.chunks_file = cls.temp_dir / "test_chunks.json"
        with open(cls.chunks_file, 'w') as f:
            json.dump({"chunks": cls.test_chunks}, f)
            
        # Initialize the RAG system
        cls.rag = RAGSystem(embedding_model=cls.model_name)
        
        # Create a larger test dataset for performance tests
        cls.large_test_chunks = []
        for i in range(100):
            cls.large_test_chunks.append({
                "text": f"Test document {i} with some content for testing purposes.",
                "metadata": {"source": f"test{i}", "chunk_index": i}
            })
        
        # Create edge case test data
        cls.edge_case_chunks = [
            {
                "text": "",  # Empty text
                "metadata": {"source": "edge1", "chunk_index": 0}
            },
            {
                "text": "   ",  # Whitespace only
                "metadata": {"source": "edge2", "chunk_index": 1}
            },
            {
                "text": "a" * 10000,  # Very long text
                "metadata": {"source": "edge3", "chunk_index": 2}
            },
            {
                "text": "Special chars: !@#$%^&*()_+{}|:\"<>?[]\\;',./~`",  # Special characters
                "metadata": {"source": "edge4", "chunk_index": 3}
            },
            {
                "text": "Unicode: 你好世界, こんにちは世界, Привет мир",  # Unicode
                "metadata": {"source": "edge5", "chunk_index": 4}
            },
            {
                "text": "The quick brown fox jumps over the lazy dog.",
                "metadata": {"source": "edge6", "chunk_index": 5, "large_field": "x" * 1000000}  # Large metadata
            }
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests are run."""
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_texts = [chunk["text"] for chunk in self.test_chunks]
        self.test_metadata = [chunk["metadata"] for chunk in self.test_chunks]
        self.edge_case_texts = [chunk["text"] for chunk in self.edge_case_chunks]
        self.edge_case_metadata = [chunk["metadata"] for chunk in self.edge_case_chunks]

    def test_process_batch(self):
        """Test the process_batch function."""
        model = SentenceTransformer(self.model_name)
        
        # Test with valid texts
        embeddings = process_batch(self.test_texts, model)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(self.test_texts))
        
        # Test with empty texts
        with self.assertRaises(ValueError):
            process_batch([], model)
            
        # Test with invalid texts
        with self.assertRaises(ValueError):
            process_batch([None, "", 123], model)

    def test_process_group(self):
        """Test the process_group function."""
        # Test with valid inputs
        embeddings, filtered_metadata = process_group(
            1, self.test_texts, self.test_metadata, self.model_name
        )
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(self.test_texts))
        self.assertEqual(len(filtered_metadata), len(self.test_metadata))
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            process_group(1, self.test_texts, self.test_metadata[:-1], self.model_name)
            
        # Test with invalid metadata
        with self.assertRaises(ValueError):
            process_group(1, self.test_texts, [None] * len(self.test_texts), self.model_name)

    def test_rag_system_retrieval(self):
        """Test the RAG system's retrieval functionality."""
        # Build a small index with test data
        texts = [chunk["text"] for chunk in self.test_chunks]
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # Create and save a test index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        index_file = self.temp_dir / "test_index.faiss"
        metadata_file = self.temp_dir / "test_metadata.pkl"
        
        faiss.write_index(index, str(index_file))
        with open(metadata_file, 'wb') as f:
            import pickle
            pickle.dump(self.test_chunks, f)
            
        # Test retrieval
        self.rag.index = index
        self.rag.metadata = self.test_chunks
        
        # Test exact match
        results = self.rag.retrieve("quick brown fox", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("quick brown", results[0]["text"].lower())
        
        # Test semantic similarity
        results = self.rag.retrieve("fast animal jumping", k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("jump" in r["text"].lower() for r in results))
        
        # Test with minimum score threshold
        results = self.rag.retrieve("completely unrelated query", k=1, min_score=0.9)
        self.assertEqual(len(results), 0)

    def test_metadata_handling(self):
        """Test proper handling of metadata throughout the system."""
        # Test metadata preservation in process_group
        embeddings, filtered_metadata = process_group(
            1, self.test_texts, self.test_metadata, self.model_name
        )
        
        # Verify metadata structure
        self.assertEqual(len(filtered_metadata), len(self.test_metadata))
        for original, filtered in zip(self.test_metadata, filtered_metadata):
            self.assertEqual(original["source"], filtered["source"])
            self.assertEqual(original["chunk_index"], filtered["chunk_index"])
            
        # Set up index for retrieval testing
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(self.test_texts, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Test metadata handling in retrieval
        self.rag.index = index
        self.rag.metadata = self.test_chunks
        results = self.rag.retrieve("quick brown fox", k=1)
        
        self.assertEqual(len(results), 1)
        self.assertTrue(all("metadata" in r for r in results))
        self.assertTrue(all("source" in r["metadata"] for r in results))
        self.assertTrue(all("chunk_index" in r["metadata"] for r in results))
        
    def test_error_recovery_model_loading(self):
        """Test system behavior when model loading fails."""
        # Test with non-existent model
        # Note: SentenceTransformer creates a default model with invalid names
        # so we'll test with a more extreme case
        with self.assertRaises(Exception):
            # This should fail as it's not a valid model architecture
            SentenceTransformer("invalid/model/name/that/does/not/exist/at/all")
            
        # Test recovery after model loading failure
        try:
            model = SentenceTransformer(self.model_name)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to recover from model loading error: {str(e)}")
            
    def test_error_recovery_corrupted_index(self):
        """Test recovery from corrupted index files."""
        # Create a corrupted index file
        corrupted_index_file = self.temp_dir / "corrupted_index.faiss"
        with open(corrupted_index_file, 'w') as f:
            f.write("This is not a valid FAISS index file")
            
        # Try to load the corrupted index
        with self.assertRaises(Exception):
            faiss.read_index(str(corrupted_index_file))
            
        # Test recovery by creating a new index
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(self.test_texts, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save the new index
        faiss.write_index(index, str(corrupted_index_file))
        
        # Verify the new index can be loaded
        try:
            loaded_index = faiss.read_index(str(corrupted_index_file))
            self.assertIsNotNone(loaded_index)
        except Exception as e:
            self.fail(f"Failed to recover from corrupted index: {str(e)}")
            
    def test_error_recovery_network_errors(self):
        """Test handling of network errors during model downloads."""
        # This is a bit tricky to test directly, so we'll simulate it
        # by temporarily changing the model name to a non-existent one
        # and then restoring it
        
        # Save original model name
        original_model_name = self.model_name
        
        # Change to non-existent model
        self.model_name = "non_existent_model_name"
        
        # Try to load the model (should fail)
        with self.assertRaises(Exception):
            SentenceTransformer(self.model_name)
            
        # Restore original model name
        self.model_name = original_model_name
        
        # Verify we can still load the original model
        try:
            model = SentenceTransformer(self.model_name)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to recover from network error: {str(e)}")
            
    def test_performance_batch_sizes(self):
        """Test processing time for different batch sizes."""
        model = SentenceTransformer(self.model_name)
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        processing_times = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            embeddings = model.encode(self.test_texts, batch_size=batch_size, normalize_embeddings=True)
            end_time = time.time()
            processing_times[batch_size] = end_time - start_time
            
        # Log the results
        logger.info(f"Processing times for different batch sizes: {processing_times}")
        
        # Verify that processing completed successfully
        self.assertTrue(all(time > 0 for time in processing_times.values()))
        
    def test_performance_memory_usage(self):
        """Test memory usage during large batch processing."""
        # Get initial memory usage
        initial_memory = get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory:.2f} GB")
        
        # Process a larger batch
        model = SentenceTransformer(self.model_name)
        large_texts = [chunk["text"] for chunk in self.large_test_chunks]
        
        # Process the batch
        embeddings = model.encode(large_texts, normalize_embeddings=True)
        
        # Get memory usage after processing
        after_memory = get_memory_usage()
        logger.info(f"Memory usage after processing: {after_memory:.2f} GB")
        
        # Clear memory
        clear_memory()
        
        # Get memory usage after clearing
        final_memory = get_memory_usage()
        logger.info(f"Memory usage after clearing: {final_memory:.2f} GB")
        
        # Verify that memory was cleared or stayed the same
        # Note: On some systems, memory might not be immediately released
        self.assertLessEqual(final_memory, after_memory + 0.1)  # Allow for small fluctuations
        
    def test_performance_retrieval_latency(self):
        """Test retrieval latency with different index sizes."""
        model = SentenceTransformer(self.model_name)
        
        # Create indices of different sizes
        small_texts = self.test_texts
        medium_texts = [chunk["text"] for chunk in self.large_test_chunks[:20]]
        large_texts = [chunk["text"] for chunk in self.large_test_chunks]
        
        # Generate embeddings
        small_embeddings = model.encode(small_texts, normalize_embeddings=True)
        medium_embeddings = model.encode(medium_texts, normalize_embeddings=True)
        large_embeddings = model.encode(large_texts, normalize_embeddings=True)
        
        # Create indices
        dimension = small_embeddings.shape[1]
        small_index = faiss.IndexFlatIP(dimension)
        medium_index = faiss.IndexFlatIP(dimension)
        large_index = faiss.IndexFlatIP(dimension)
        
        small_index.add(small_embeddings)
        medium_index.add(medium_embeddings)
        large_index.add(large_embeddings)
        
        # Test retrieval latency
        query = "test query"
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # Measure retrieval time for each index size
        retrieval_times = {}
        
        # Small index
        start_time = time.time()
        small_index.search(query_embedding, 5)
        end_time = time.time()
        retrieval_times["small"] = end_time - start_time
        
        # Medium index
        start_time = time.time()
        medium_index.search(query_embedding, 5)
        end_time = time.time()
        retrieval_times["medium"] = end_time - start_time
        
        # Large index
        start_time = time.time()
        large_index.search(query_embedding, 5)
        end_time = time.time()
        retrieval_times["large"] = end_time - start_time
        
        # Log the results
        logger.info(f"Retrieval times for different index sizes: {retrieval_times}")
        
        # Verify that retrieval completed successfully
        self.assertTrue(all(time > 0 for time in retrieval_times.values()))
        
    def test_edge_cases_long_texts(self):
        """Test with very long texts."""
        # Create a very long text
        long_text = "This is a test sentence. " * 1000
        
        # Process the long text
        model = SentenceTransformer(self.model_name)
        embeddings = process_batch([long_text], model)
        
        # Verify that the embedding was generated
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), 1)
        
    def test_edge_cases_special_characters(self):
        """Test with special characters and Unicode."""
        # Process texts with special characters
        model = SentenceTransformer(self.model_name)
        
        # Filter out empty and whitespace-only texts
        valid_texts = [text for text in self.edge_case_texts if text.strip()]
        
        # Process the texts
        embeddings = process_batch(valid_texts, model)
        
        # Verify that embeddings were generated
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(valid_texts))
        
    def test_edge_cases_empty_texts(self):
        """Test with empty or whitespace-only texts."""
        # Process texts including empty and whitespace-only
        model = SentenceTransformer(self.model_name)
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            process_batch(self.edge_case_texts[:2], model)  # First two are empty/whitespace
            
    def test_edge_cases_large_metadata(self):
        """Test with extremely large metadata objects."""
        # Process texts with large metadata
        model = SentenceTransformer(self.model_name)
        
        # Filter out empty and whitespace-only texts
        valid_texts = [text for text in self.edge_case_texts if text.strip()]
        valid_metadata = [meta for i, meta in enumerate(self.edge_case_metadata) if self.edge_case_texts[i].strip()]
        
        # Process the texts
        embeddings, filtered_metadata = process_group(1, valid_texts, valid_metadata, self.model_name)
        
        # Verify that embeddings were generated
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(valid_texts))
        self.assertEqual(len(filtered_metadata), len(valid_metadata))
        
    def test_integration_full_pipeline(self):
        """Test the full pipeline from document ingestion to retrieval."""
        # Create a temporary RAG system
        rag = RAGSystem(embedding_model=self.model_name)
        
        # Process documents
        chunks = rag.process_documents([chunk["text"] for chunk in self.test_chunks])
        
        # Verify chunks were created
        self.assertEqual(len(chunks), len(self.test_chunks))
        
        # Build index
        rag.build_index(chunks)
        
        # Verify index was built
        self.assertIsNotNone(rag.index)
        
        # Test retrieval
        results = rag.retrieve("quick brown fox", k=1)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertIn("quick brown", results[0]["text"].lower())
        
    def test_integration_configuration(self):
        """Test configuration loading and validation."""
        # Create a RAG system with custom configuration
        rag = RAGSystem(
            embedding_model=self.model_name,
            chunk_size=256,
            faiss_type="flat",
            nlist=50,
            batch_size=16
        )
        
        # Verify configuration was applied
        self.assertEqual(rag.chunk_size, 256)
        self.assertEqual(rag.faiss_type, "flat")
        self.assertEqual(rag.nlist, 50)
        self.assertEqual(rag.batch_size, 16)
        
        # Test with invalid configuration - use a non-existent model
        with self.assertRaises(Exception):
            RAGSystem(embedding_model="invalid/model/name/that/does/not/exist/at/all")
            
    def test_integration_external_systems(self):
        """Test interaction with external systems."""
        # This is a placeholder for testing interaction with external systems
        # In a real system, this would test API calls, database interactions, etc.
        
        # For now, we'll just verify that the RAG system can be initialized
        rag = RAGSystem(embedding_model=self.model_name)
        self.assertIsNotNone(rag)
        
        # Test saving and loading index
        # Create a test index
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(self.test_texts, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save the index
        index_file = self.temp_dir / "test_index.faiss"
        faiss.write_index(index, str(index_file))
        
        # Load the index
        loaded_index = faiss.read_index(str(index_file))
        
        # Verify the index was loaded correctly
        self.assertIsNotNone(loaded_index)
        
        # Test query on loaded index
        query = "quick brown fox"
        query_embedding = model.encode([query], normalize_embeddings=True)
        scores, indices = loaded_index.search(query_embedding, 1)
        
        # Verify the search worked
        self.assertEqual(len(indices[0]), 1)
        self.assertEqual(len(scores[0]), 1)

if __name__ == '__main__':
    unittest.main() 