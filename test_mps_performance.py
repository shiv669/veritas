#!/usr/bin/env python3
"""
test_mps_performance.py

A script to test and benchmark the performance of the RAG system with MPS optimizations.
This script compares the performance of different device configurations.
"""

import json
import logging
import time
import torch
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    METADATA_FILE,
    DEFAULT_GEN_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DEVICE
)
from veritas.mps_utils import (
    get_optimal_device,
    optimize_for_mps,
    get_optimal_batch_size,
    prepare_inputs_for_mps,
    get_memory_info,
    log_memory_info
)

def benchmark_embedding_generation(device: str, batch_size: int, num_samples: int = 100):
    """Benchmark embedding generation performance."""
    logger.info(f"Benchmarking embedding generation on {device} with batch size {batch_size}")
    
    # Initialize RAG system
    rag = RAGSystem(device=device)
    
    # Generate sample texts
    sample_texts = [f"This is sample text {i} for benchmarking embedding generation." for i in range(num_samples)]
    
    # Measure time
    start_time = time.time()
    
    # Generate embeddings
    embeddings = rag.generate_embeddings(sample_texts)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    embeddings_per_second = num_samples / elapsed_time
    
    logger.info(f"Generated {num_samples} embeddings in {elapsed_time:.2f} seconds")
    logger.info(f"Performance: {embeddings_per_second:.2f} embeddings/second")
    
    return {
        "device": device,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "elapsed_time": elapsed_time,
        "embeddings_per_second": embeddings_per_second
    }

def benchmark_model_inference(device: str, model_name: str = DEFAULT_GEN_MODEL, num_samples: int = 10):
    """Benchmark model inference performance."""
    logger.info(f"Benchmarking model inference on {device} with model {model_name}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure model based on device
    if device == "mps":
        # MPS-specific configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,  # Don't use device_map with MPS
            torch_dtype=torch.float32  # MPS works better with float32
        )
        model = model.to(device)
    elif device == "cpu":
        # CPU-specific configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,  # Don't use device_map with CPU
            torch_dtype=torch.float32
        )
        model = model.to("cpu")
    else:
        # Standard configuration for other devices
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    # Generate sample prompts
    sample_prompts = [
        "Explain the concept of RAG (Retrieval-Augmented Generation) in simple terms.",
        "What are the benefits of using FAISS for vector search?",
        "How do sentence transformers help in semantic search?",
        "Explain the difference between dense and sparse retrievers.",
        "What is the role of chunking in RAG systems?",
        "How does the retrieval process work in a RAG system?",
        "What are the limitations of RAG systems?",
        "How can you improve the quality of retrieved documents?",
        "Explain the concept of hybrid search in RAG systems.",
        "What are the best practices for implementing a RAG system?"
    ]
    
    # Ensure we don't exceed the number of samples
    sample_prompts = sample_prompts[:num_samples]
    
    # Measure time
    start_time = time.time()
    
    # Generate responses
    for prompt in tqdm(sample_prompts):
        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set attention mask for better results
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        # Generate response
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,  # Keep it short for benchmarking
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        # Decode response (not needed for benchmarking, but included for completeness)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    tokens_per_second = sum(len(outputs[0]) for _ in sample_prompts) / elapsed_time
    
    logger.info(f"Generated {len(sample_prompts)} responses in {elapsed_time:.2f} seconds")
    logger.info(f"Performance: {tokens_per_second:.2f} tokens/second")
    
    return {
        "device": device,
        "model": model_name,
        "num_samples": len(sample_prompts),
        "elapsed_time": elapsed_time,
        "tokens_per_second": tokens_per_second
    }

def main():
    """Run performance benchmarks for different device configurations."""
    parser = argparse.ArgumentParser(description="Benchmark RAG system performance with different device configurations")
    parser.add_argument("--device", type=str, default=None, help="Device to use (mps, cpu, or auto)")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--model-batch-size", type=int, default=1, help="Batch size for model inference")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for benchmarking")
    parser.add_argument("--model", type=str, default=DEFAULT_GEN_MODEL, help="Model to use for inference")
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = get_optimal_device()
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    log_memory_info()
    
    # Run benchmarks
    embedding_results = benchmark_embedding_generation(
        device=device,
        batch_size=args.embedding_batch_size,
        num_samples=args.num_samples
    )
    
    inference_results = benchmark_model_inference(
        device=device,
        model_name=args.model,
        num_samples=min(10, args.num_samples)  # Limit to 10 for model inference
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Performance Benchmark Results")
    logger.info("="*50)
    logger.info(f"Device: {device}")
    logger.info(f"Embedding Generation: {embedding_results['embeddings_per_second']:.2f} embeddings/second")
    logger.info(f"Model Inference: {inference_results['tokens_per_second']:.2f} tokens/second")
    logger.info("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 