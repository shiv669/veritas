#!/usr/bin/env python3
"""
index_chunks_parallel.py

Script to index the processed chunks into the RAG system.
This script loads the chunks from data/chunks.json and builds a FAISS index.
Optimized for M4 Max with Apple Silicon, 120GB RAM, and 8TB storage.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import torch
import gc
import psutil
import time
import traceback
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
import platform
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import argparse

# Set memory optimization environment variables before importing PyTorch-related modules
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable automatic growth
os.environ["PYTORCH_MPS_MEMORY_LIMIT"] = "102GB"  # 80% of 128GB
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root to Python path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.veritas.rag import RAGSystem
from src.veritas.config import Config
from src.veritas.utils import setup_logging
from src.veritas.mps_utils import is_mps_available, optimize_for_mps, optimize_memory_for_m4

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Index chunks for RAG system")
parser.add_argument("--input", type=str, default="data/chunks/chunked_data.json", 
                    help="Path to input chunks file")
parser.add_argument("--output", type=str, default=None,
                    help="Output directory for index (defaults to timestamped directory)")
parser.add_argument("--batch-size", type=int, default=16,
                    help="Batch size for processing")
parser.add_argument("--group-size", type=int, default=10,
                    help="Group size for chunk processing")
parser.add_argument("--device", type=str, default=None,
                    help="Device to use (mps, cuda, or cpu)")
parser.add_argument("--max-chunks", type=int, default=None,
                    help="Maximum number of chunks to process")
args = parser.parse_args()

# Configure logging
logger = setup_logging(__name__)

# M4 Max specific optimizations
IS_APPLE_SILICON = platform.processor() == 'arm'
NUM_CORES = mp.cpu_count()
NUM_WORKERS = max(1, NUM_CORES - 4)  # Leave more cores free for system
BATCH_SIZE = args.batch_size  # Use command-line value
GROUP_SIZE = args.group_size  # Use command-line value
MEMORY_THRESHOLD = 0.7 * psutil.virtual_memory().total  # 70% of total RAM

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

def should_clear_memory():
    """Check if memory usage is above threshold."""
    return psutil.virtual_memory().available < (MEMORY_THRESHOLD * 0.2)

def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
    gc.collect()

def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate that required metadata fields are present."""
    required_fields = ['source', 'chunk_id', 'text']
    return all(field in metadata for field in required_fields)

def process_chunk(chunk: Dict[str, Any], embedding_model: SentenceTransformer) -> Dict[str, Any]:
    """
    Process a single chunk by computing its embedding.
    
    Args:
        chunk: Chunk dictionary
        embedding_model: SentenceTransformer model
        
    Returns:
        Dictionary with chunk data and embedding
    """
    try:
        # Get the text, checking both 'text' and 'content' fields
        text = chunk.get('text', '')
        
        # Try alternate fields if text is empty
        if not text:
            # Try content field
            text = chunk.get('content', '')
            
        # If still empty, check if there's a nested metadata field
        if not text and 'metadata' in chunk:
            text = chunk['metadata'].get('text', '')
            
        # Log content keys to help diagnose
        if not text:
            logger.warning(f"Empty text for chunk {chunk.get('chunk_id', 'unknown')}. Available keys: {list(chunk.keys())}")
            if len(chunk.keys()) < 5:  # Only log full content for small chunks
                logger.debug(f"Full chunk content: {chunk}")
            return None
            
        # Compute embedding
        embedding = embedding_model.encode(text, normalize_embeddings=True)
        
        # Add embedding to chunk
        result = {
            'chunk_id': chunk.get('chunk_id', ''),
            'source': chunk.get('source', ''),
            'text': text,
            'embedding': embedding,
            **{k: v for k, v in chunk.items() if k not in ['text', 'source', 'chunk_id']}
        }
        
        return result
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_chunk_group(chunks: List[Dict[str, Any]], embedding_model: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    Process a group of chunks.
    
    Args:
        chunks: List of chunks to process
        embedding_model: SentenceTransformer model
        
    Returns:
        List of processed chunks with embeddings
    """
    processed_chunks = []
    
    for chunk in chunks:
        processed = process_chunk(chunk, embedding_model)
        if processed:
            processed_chunks.append(processed)
            
        # Clear memory if needed
        if should_clear_memory():
            clear_memory()
    
    return processed_chunks

def build_faiss_index(processed_chunks: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Build a FAISS index from processed chunks.
    
    Args:
        processed_chunks: List of processed chunks with embeddings
        
    Returns:
        Tuple of (FAISS index, chunks without embeddings)
    """
    # Extract embeddings and metadata
    embeddings = []
    chunks_without_embeddings = []
    
    # Check if we have any processed chunks
    if not processed_chunks:
        logger.error("No valid chunks to build index from")
        # Create a dummy index to avoid errors
        dimension = 384  # Default dimension for all-MiniLM-L6-v2
        index = faiss.IndexFlatIP(dimension)
        return index, []
    
    for chunk in processed_chunks:
        embeddings.append(chunk['embedding'])
        # Create a copy without the embedding to save memory
        chunk_without_embedding = {k: v for k, v in chunk.items() if k != 'embedding'}
        chunks_without_embeddings.append(chunk_without_embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Check if we have any embeddings
    if len(embeddings) == 0:
        logger.error("No embeddings found in processed chunks")
        # Create a dummy index to avoid errors
        dimension = 384  # Default dimension for all-MiniLM-L6-v2
        index = faiss.IndexFlatIP(dimension)
        return index, chunks_without_embeddings
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    return index, chunks_without_embeddings

def save_index_and_chunks(index: faiss.Index, chunks: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save FAISS index and chunks to disk.
    
    Args:
        index: FAISS index
        chunks: Chunks metadata
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save index
    index_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, index_path)
    logger.info(f"Saved FAISS index to {index_path}")
    
    # Save chunks
    chunks_path = os.path.join(output_dir, "chunks.json")
    with open(chunks_path, 'w') as f:
        json.dump(chunks, f)
    logger.info(f"Saved chunks to {chunks_path}")

def main():
    """Main function to index chunks into the RAG system."""
    try:
        # Apply memory optimizations for M4
        optimize_memory_for_m4()
        logger.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
        
        # Choose the appropriate device
        device = args.device
        if device is None:
            device = "mps" if is_mps_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load the embedding model
        embedding_model_name = Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        
        if device == "mps":
            # Optimize for MPS
            embedding_model.to(device)
            logger.info("Moved embedding model to MPS device")

        # Load chunks from input file
        input_file = Path(args.input)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Output directory
        if args.output:
            output_dir = args.output
        else:
            output_dir = os.path.join(Config.INDICES_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Process chunks in smaller groups
        logger.info(f"Processing chunks from {input_file}")
        with open(input_file, 'r') as f:
            chunks = json.load(f)
            
            # Check chunk format
            if not isinstance(chunks, list):
                logger.info(f"Chunk file format is not a list. Type: {type(chunks)}")
                # If it's a dict, convert appropriately
                if isinstance(chunks, dict):
                    # Check for common formats
                    if 'documents' in chunks:
                        chunks = chunks['documents']
                        logger.info(f"Extracted {len(chunks)} documents from chunks")
                    elif 'chunks' in chunks:
                        chunks = chunks['chunks']
                        logger.info(f"Extracted {len(chunks)} chunks from 'chunks' key")
                    else:
                        # Last resort - convert dict items to list
                        chunks = [{"chunk_id": k, **v} for k, v in chunks.items()]
                        logger.info(f"Converted dictionary to {len(chunks)} chunks")
                else:
                    # Wrap in a list if it's something else
                    chunks = [chunks]
            
            # Apply max_chunks limit if specified
            if args.max_chunks and len(chunks) > args.max_chunks:
                logger.info(f"Limiting to {args.max_chunks} chunks (out of {len(chunks)})")
                chunks = chunks[:args.max_chunks]
                
            total_chunks = len(chunks)
            logger.info(f"Total chunks to process: {total_chunks}")
            
            # Log a sample chunk for debugging
            if chunks:
                sample_chunk = chunks[0]
                logger.info(f"Sample chunk format: {list(sample_chunk.keys())}")
                if len(sample_chunk.keys()) < 10:  # Only log full content for small chunks
                    logger.info(f"Sample chunk content: {sample_chunk}")
            
            # Process all chunks
            all_processed_chunks = []
            
            # Process chunks in groups
            for i in range(0, len(chunks), GROUP_SIZE):
                group = chunks[i:i + GROUP_SIZE]
                processed_group = process_chunk_group(group, embedding_model)
                all_processed_chunks.extend(processed_group)
                
                # Log progress and success rate
                if i % (GROUP_SIZE * 10) == 0 or i + GROUP_SIZE >= len(chunks):
                    logger.info(f"Processed {i + len(group)}/{total_chunks} chunks")
                    logger.info(f"Memory usage: {get_memory_usage():.2f} GB")
                    success_rate = len(all_processed_chunks) / (i + len(group)) * 100 if i + len(group) > 0 else 0
                    logger.info(f"Successfully processed chunks: {len(all_processed_chunks)} ({success_rate:.2f}%)")
                
                # Clear memory after each group
                if should_clear_memory():
                    clear_memory()
            
            logger.info(f"Building FAISS index from {len(all_processed_chunks)} processed chunks")
            
            # Only build index if we have processed chunks
            if all_processed_chunks:
                index, chunks_without_embeddings = build_faiss_index(all_processed_chunks)
                
                # Save results
                logger.info(f"Saving results to {output_dir}")
                save_index_and_chunks(index, chunks_without_embeddings, output_dir)
                
                logger.info("Indexing complete!")
            else:
                logger.error("No chunks were successfully processed. Check the JSON format and data.")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 