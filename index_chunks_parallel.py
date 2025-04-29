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
import tempfile
import shutil
import ijson
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
import platform

from veritas.rag import RAGSystem
from veritas.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAISS_TYPE,
    DEFAULT_NLIST,
    DEFAULT_BATCH_SIZE,
    FAISS_INDEX_FILE,
    METADATA_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the chunks file path
CHUNKS_FILE = Path("data/chunks.json")
TEMP_DIR = Path("data/temp_embeddings")

# M4 Max specific optimizations
IS_APPLE_SILICON = platform.processor() == 'arm'
NUM_CORES = mp.cpu_count()
# Leave 2 cores free for system processes
NUM_WORKERS = max(1, NUM_CORES - 2)
# Larger batch size for M4 Max with 120GB RAM
OPTIMIZED_BATCH_SIZE = 64
# Larger group size for processing
GROUP_SIZE = 100
# Memory threshold (90% of available RAM)
MEMORY_THRESHOLD = 0.9 * (120 * 1024 * 1024 * 1024)  # 90% of 120GB in bytes

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_available_memory():
    """Get available memory in bytes."""
    return psutil.virtual_memory().available

def should_clear_memory():
    """Check if memory usage is above threshold."""
    return get_available_memory() < (MEMORY_THRESHOLD * 0.1)  # 10% of threshold

def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
    gc.collect()
    
    # Force garbage collection of numpy arrays
    for name in list(globals()):
        if name.startswith('_') or name in ['np', 'torch', 'gc', 'psutil']:
            continue
        try:
            obj = globals()[name]
            if isinstance(obj, np.ndarray):
                del globals()[name]
        except:
            pass
    gc.collect()

def process_batch(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Process a batch of texts to generate embeddings.
    
    Args:
        texts: List of texts to process
        model: SentenceTransformer model to use
        
    Returns:
        numpy array of embeddings
        
    Raises:
        ValueError: If texts list is empty or contains invalid texts
        RuntimeError: If encoding fails
    """
    if not texts:
        logging.warning("Empty text list provided to process_batch")
        raise ValueError("Empty text list provided")
        
    # Filter out empty or invalid texts
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < len(texts):
        logging.warning(f"Filtered out {len(texts) - len(valid_texts)} invalid texts")
    
    if not valid_texts:
        logging.warning("No valid texts to process after filtering")
        raise ValueError("No valid texts to process")
        
    try:
        logging.debug(f"Processing batch of {len(valid_texts)} texts")
        
        # Use MPS (Metal Performance Shaders) for Apple Silicon if available
        if IS_APPLE_SILICON and torch.backends.mps.is_available():
            device = torch.device("mps")
            model.to(device)
            logging.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        else:
            device = torch.device("cpu")
            model.to(device)
            logging.info("Using CPU for processing")
            
        # Process in smaller sub-batches if needed
        if len(valid_texts) > OPTIMIZED_BATCH_SIZE:
            sub_batches = [valid_texts[i:i + OPTIMIZED_BATCH_SIZE] for i in range(0, len(valid_texts), OPTIMIZED_BATCH_SIZE)]
            embeddings_list = []
            for sub_batch in sub_batches:
                sub_embeddings = model.encode(sub_batch, convert_to_numpy=True)
                embeddings_list.append(sub_embeddings)
                # Clear memory after each sub-batch
                if should_clear_memory():
                    clear_memory()
            embeddings = np.vstack(embeddings_list)
        else:
            embeddings = model.encode(valid_texts, convert_to_numpy=True)
            
        if embeddings.shape[0] != len(valid_texts):
            raise RuntimeError(f"Mismatch between embeddings ({embeddings.shape[0]}) and texts ({len(valid_texts)})")
        return embeddings
        
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Failed to process batch: {str(e)}")

def save_embeddings(embeddings, file_path):
    """Save embeddings to a file."""
    try:
        np.save(file_path, embeddings)
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
        return False
    return True

def load_embeddings(file_path):
    """Load embeddings from a file."""
    try:
        return np.load(file_path)
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        return None

def count_chunks():
    """Count the number of chunks in the JSON file."""
    count = 0
    try:
        with open(CHUNKS_FILE, 'rb') as f:
            parser = ijson.parse(f)
            for prefix, event, value in parser:
                if prefix.endswith('.text'):
                    count += 1
    except Exception as e:
        logger.error(f"Error counting chunks: {str(e)}")
        return 0
    return count

def chunk_generator():
    """Generate chunks from the JSON file."""
    current_chunk = None
    try:
        with open(CHUNKS_FILE, 'rb') as f:
            parser = ijson.parse(f)
            for prefix, event, value in parser:
                if prefix.endswith('.text'):
                    if current_chunk is not None:
                        yield current_chunk
                    current_chunk = {"text": value}
                elif prefix.endswith('.metadata'):
                    if current_chunk is not None:
                        current_chunk["metadata"] = value
        if current_chunk is not None:
            yield current_chunk
    except Exception as e:
        logger.error(f"Error in chunk generator: {str(e)}")
        return

def chunks_generator(batch_size: int = 1000):
    """Generate chunks from the JSON file in batches.
    
    Args:
        batch_size: Number of chunks to yield at once
        
    Yields:
        List of chunk dictionaries
    """
    current_batch = []
    
    try:
        with open(CHUNKS_FILE, 'rb') as f:
            parser = ijson.parse(f)
            current_chunk = None
            
            for prefix, event, value in parser:
                if prefix.endswith('.text'):
                    if current_chunk is not None:
                        current_batch.append(current_chunk)
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                            clear_memory()
                    current_chunk = {"text": value}
                elif prefix.endswith('.metadata'):
                    if current_chunk is not None:
                        current_chunk["metadata"] = value
            
            # Add the last chunk if it exists
            if current_chunk is not None:
                current_batch.append(current_chunk)
                
            # Yield the remaining chunks
            if current_batch:
                yield current_batch
                
    except Exception as e:
        logger.error(f"Error in chunks_generator: {str(e)}")
        logger.error(traceback.format_exc())
        return

def process_group(group_id: int, texts: List[str], metadata: List[Dict], model_name: str) -> Tuple[np.ndarray, List[Dict]]:
    """Process a group of texts and their metadata.
    
    Args:
        group_id: ID of the processing group
        texts: List of texts to process
        metadata: List of metadata dictionaries
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        Tuple of (embeddings array, filtered metadata list)
        
    Raises:
        RuntimeError: If processing fails
    """
    if len(texts) != len(metadata):
        logging.error(f"Group {group_id}: Mismatch between texts ({len(texts)}) and metadata ({len(metadata)})")
        raise ValueError("Texts and metadata length mismatch")
        
    try:
        logging.info(f"Group {group_id}: Loading model {model_name}")
        
        # Use MPS (Metal Performance Shaders) for Apple Silicon if available
        if IS_APPLE_SILICON and torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU for processing")
            
        model = SentenceTransformer(model_name)
        model.to(device)
        
        # Filter out empty or invalid texts while keeping track of indices
        valid_pairs = [(i, text) for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
        if not valid_pairs:
            logging.warning(f"Group {group_id}: No valid texts to process")
            return np.array([]), []
            
        valid_indices, valid_texts = zip(*valid_pairs)
        logging.info(f"Group {group_id}: Processing {len(valid_texts)} valid texts out of {len(texts)} total")
        
        # Process texts in batches
        embeddings = process_batch(list(valid_texts), model)
        
        # Clear GPU memory
        del model
        if IS_APPLE_SILICON and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Filter metadata to match valid texts
        filtered_metadata = [metadata[i] for i in valid_indices]
        
        if len(embeddings) != len(filtered_metadata):
            raise RuntimeError(f"Group {group_id}: Mismatch between embeddings ({len(embeddings)}) and metadata ({len(filtered_metadata)})")
            
        logging.info(f"Group {group_id}: Successfully processed {len(embeddings)} texts")
        return embeddings, filtered_metadata
        
    except Exception as e:
        logging.error(f"Group {group_id}: Processing failed - {str(e)}")
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Failed to process group {group_id}: {str(e)}")

def main():
    """Main function to index chunks into the RAG system."""
    try:
        # Check if chunks file exists
        if not CHUNKS_FILE.exists():
            logger.error(f"Chunks file not found: {CHUNKS_FILE}")
            return
        
        # Create temp directory
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Count total chunks
        logger.info("Counting chunks...")
        total_chunks = count_chunks()
        if total_chunks == 0:
            logger.error("No chunks found or error counting chunks")
            return
            
        logger.info(f"Found {total_chunks} chunks")
        logger.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
        logger.info(f"Using {NUM_WORKERS} workers on {NUM_CORES} cores")
        
        # Process chunks in groups
        temp_files = []
        current_group = []
        group_count = 0
        
        logger.info("Processing chunks...")
        with mp.Pool(NUM_WORKERS) as pool:
            for chunk_batch in chunks_generator(batch_size=GROUP_SIZE):
                current_group.extend(chunk_batch)
                
                if len(current_group) >= GROUP_SIZE:
                    group_count += 1
                    logger.info(f"Processing group {group_count}")
                    logger.info(f"Memory usage before group: {get_memory_usage():.2f} GB")
                    
                    # Process the group
                    result = process_group(group_count, [chunk.get('text', '') for chunk in current_group], 
                                         [chunk.get('metadata', {}) for chunk in current_group], 
                                         DEFAULT_EMBEDDING_MODEL)
                    
                    if result[0].size > 0:
                        # Save embeddings and metadata
                        try:
                            temp_file = TEMP_DIR / f"embeddings_group_{group_count}.npy"
                            if save_embeddings(result[0], temp_file):
                                temp_files.append(temp_file)
                                
                            metadata_file = TEMP_DIR / f"metadata_group_{group_count}.json"
                            with open(metadata_file, 'w') as f:
                                json.dump(result[1], f)
                        except Exception as e:
                            logger.error(f"Error saving group {group_count}: {e}")
                    
                    current_group = []
                    clear_memory()
                    logger.info(f"Memory usage after group: {get_memory_usage():.2f} GB")
        
        # Process remaining chunks if any
        if current_group:
            group_count += 1
            logger.info(f"Processing final group {group_count}")
            
            result = process_group(group_count, [chunk.get('text', '') for chunk in current_group], 
                                 [chunk.get('metadata', {}) for chunk in current_group], 
                                 DEFAULT_EMBEDDING_MODEL)
            if result[0].size > 0:
                temp_file = TEMP_DIR / f"embeddings_group_{group_count}.npy"
                if save_embeddings(result[0], temp_file):
                    temp_files.append(temp_file)
                    
                    metadata_file = TEMP_DIR / f"metadata_group_{group_count}.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(result[1], f)
        
        if not temp_files:
            logger.error("No embeddings were successfully generated")
            return
            
        # Initialize FAISS index using first group
        logger.info("Loading first group to get dimensions")
        first_embeddings = load_embeddings(temp_files[0])
        if first_embeddings is None:
            logger.error("Could not load first group embeddings")
            return
            
        dimension = first_embeddings.shape[1]
        
        logger.info(f"Creating FAISS index with dimension {dimension}")
        if DEFAULT_FAISS_TYPE == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif DEFAULT_FAISS_TYPE == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            train_nlist = min(DEFAULT_NLIST, total_chunks // 10)
            index = faiss.IndexIVFFlat(quantizer, dimension, train_nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train on first group
            logger.info("Training IVF index on first group")
            index.train(first_embeddings)
        
        # Add first group to index
        logger.info("Adding first group to index")
        for i in range(0, len(first_embeddings), OPTIMIZED_BATCH_SIZE):
            chunk = first_embeddings[i:i + OPTIMIZED_BATCH_SIZE]
            index.add(chunk)
        
        # Clear memory
        del first_embeddings
        clear_memory()
        
        # Add remaining groups to index
        all_chunks = []
        try:
            with open(TEMP_DIR / "metadata_group_1.json") as f:
                all_chunks.extend(json.load(f))
        except Exception as e:
            logger.error(f"Error loading first group metadata: {str(e)}")
            return
            
        for i, temp_file in enumerate(temp_files[1:], 1):
            logger.info(f"Processing saved group {i+1}/{len(temp_files)}")
            embeddings = load_embeddings(temp_file)
            if embeddings is None:
                continue
                
            try:
                for j in range(0, len(embeddings), OPTIMIZED_BATCH_SIZE):
                    chunk = embeddings[j:j + OPTIMIZED_BATCH_SIZE]
                    index.add(chunk)
                    
                with open(TEMP_DIR / f"metadata_group_{i+1}.json") as f:
                    group_chunks = json.load(f)
                    all_chunks.extend(group_chunks)
            except Exception as e:
                logger.error(f"Error processing group {i+1}: {str(e)}")
                continue
                
            del embeddings
            clear_memory()
        
        # Save the final index and metadata
        logger.info("Saving index and metadata")
        try:
            faiss.write_index(index, str(FAISS_INDEX_FILE))
            with open(METADATA_FILE, 'w') as f:
                json.dump(all_chunks, f)
        except Exception as e:
            logger.error(f"Error saving final files: {str(e)}")
            return
            
        logger.info("Cleaning up temporary files")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
        
        logger.info("Indexing complete!")
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        logger.error(traceback.format_exc())
        
if __name__ == "__main__":
    main() 