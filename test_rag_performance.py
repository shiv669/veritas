#!/usr/bin/env python3
"""
Performance benchmarking script for the Veritas RAG system.
Tests different configurations and devices to evaluate system performance.
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
    DEFAULT_EMB_MODEL,
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
    EMBED_MAX_LENGTH,
    GEN_MAX_LENGTH
)
from veritas.mps_utils import get_memory_info, log_memory_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    device: str
    batch_size: int
    duration: float
    memory_used: float
    samples_processed: int
    throughput: float  # samples per second

class RAGBenchmark:
    """Benchmarking suite for RAG system performance."""
    
    def __init__(self, device: str = None):
        """Initialize the benchmark suite."""
        self.device = device or get_device()
        self.rag = None
        self.results = []
        
        # Test data
        self.test_documents = self._load_test_documents()
        self.test_queries = self._load_test_queries()
    
    def _load_test_documents(self) -> List[Dict]:
        """Load test documents for benchmarking."""
        doc_path = Path("test_docs/sample.txt")
        if not doc_path.exists():
            raise FileNotFoundError(f"Test document not found at {doc_path}")
        
        with open(doc_path, "r") as f:
            content = f.read()
        
        return [{
            "content": content,
            "metadata": {
                "source": "test_docs/sample.txt",
                "type": "documentation",
                "topic": "RAG systems",
                "created_at": "2024-03-21"
            }
        }]
    
    def _load_test_queries(self) -> List[str]:
        """Load test queries for benchmarking."""
        return [
            "What are the key components of a RAG system?",
            "Why is chunking important in RAG and what are the best practices?",
            "What are the main benefits of using RAG for question answering?",
            "What implementation considerations should be taken into account when building a RAG system?"
        ]
    
    def _benchmark_operation(self, operation: str, func, *args, **kwargs) -> BenchmarkResult:
        """Run a benchmark for a specific operation."""
        start_time = time.time()
        start_memory = get_memory_info()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = get_memory_info()
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        return BenchmarkResult(
            operation=operation,
            device=self.device,
            batch_size=EMBED_BATCH_SIZE if "embed" in operation.lower() else GEN_BATCH_SIZE,
            duration=duration,
            memory_used=memory_used,
            samples_processed=len(args[0]) if args else 1,
            throughput=len(args[0]) / duration if args else 1 / duration
        )
    
    def run_benchmarks(self):
        """Run all benchmarks."""
        logger.info(f"Starting benchmarks on device: {self.device}")
        log_memory_info()
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        self.rag = RAGSystem(device=self.device)
        
        # Benchmark document processing
        logger.info("\nBenchmarking document processing...")
        result = self._benchmark_operation(
            "document_processing",
            self.rag.process_documents,
            self.test_documents
        )
        self.results.append(result)
        logger.info(f"Processed {result.samples_processed} documents in {result.duration:.2f} seconds")
        logger.info(f"Throughput: {result.throughput:.2f} documents/second")
        
        # Benchmark embedding generation
        logger.info("\nBenchmarking embedding generation...")
        texts = [doc["content"] for doc in self.test_documents]
        result = self._benchmark_operation(
            "embedding_generation",
            self.rag.generate_embeddings,
            texts
        )
        self.results.append(result)
        logger.info(f"Generated {result.samples_processed} embeddings in {result.duration:.2f} seconds")
        logger.info(f"Throughput: {result.throughput:.2f} embeddings/second")
        
        # Benchmark retrieval
        logger.info("\nBenchmarking retrieval...")
        for query in tqdm(self.test_queries, desc="Testing retrieval"):
            result = self._benchmark_operation(
                "retrieval",
                self.rag.retrieve,
                query
            )
            self.results.append(result)
        
        # Benchmark response generation
        logger.info("\nBenchmarking response generation...")
        for query in tqdm(self.test_queries, desc="Testing generation"):
            context = self.rag.retrieve(query)
            result = self._benchmark_operation(
                "response_generation",
                self.rag.generate_response,
                query,
                context
            )
            self.results.append(result)
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save benchmark results to a JSON file."""
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"rag_benchmark_{timestamp}.json"
        
        results_data = {
            "device": self.device,
            "model_settings": {
                "embedding_model": DEFAULT_EMB_MODEL,
                "generation_model": DEFAULT_GEN_MODEL,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "repetition_penalty": REPETITION_PENALTY
            },
            "performance_settings": {
                "embed_batch_size": EMBED_BATCH_SIZE,
                "gen_batch_size": GEN_BATCH_SIZE,
                "embed_max_length": EMBED_MAX_LENGTH,
                "gen_max_length": GEN_MAX_LENGTH
            },
            "results": [vars(result) for result in self.results]
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"\nBenchmark results saved to: {results_file}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print a summary of benchmark results."""
        logger.info("\nBenchmark Summary:")
        logger.info(f"Device: {self.device}")
        logger.info(f"Embedding Model: {DEFAULT_EMB_MODEL}")
        logger.info(f"Generation Model: {DEFAULT_GEN_MODEL}")
        
        # Calculate averages by operation
        operation_results = {}
        for result in self.results:
            if result.operation not in operation_results:
                operation_results[result.operation] = []
            operation_results[result.operation].append(result)
        
        logger.info("\nPerformance by Operation:")
        for operation, results in operation_results.items():
            avg_duration = np.mean([r.duration for r in results])
            avg_throughput = np.mean([r.throughput for r in results])
            logger.info(f"{operation}:")
            logger.info(f"  Average Duration: {avg_duration:.2f} seconds")
            logger.info(f"  Average Throughput: {avg_throughput:.2f} samples/second")

def main():
    try:
        benchmark = RAGBenchmark()
        benchmark.run_benchmarks()
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 