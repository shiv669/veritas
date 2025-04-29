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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import sys

# Add the project root to Python path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.veritas.rag import RAGSystem
from src.veritas.config import Config
from src.veritas.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)

# Define the chunks file path
CHUNKS_FILE = Path(os.path.join(Config.CHUNKS_DIR, "chunked_data.json"))
TEMP_DIR = Path(os.path.join(Config.DATA_DIR, "temp_embeddings"))
FAISS_INDEX_FILE = os.path.join(Config.INDICES_DIR, "index.faiss")
METADATA_FILE = os.path.join(Config.INDICES_DIR, "metadata.json")

# Default settings
DEFAULT_EMBEDDING_MODEL = Config.EMBEDDING_MODEL
DEFAULT_FAISS_TYPE = "IndexFlatIP"  # Inner product (cosine similarity)
DEFAULT_NLIST = 100  # Number of clusters for IVF
DEFAULT_BATCH_SIZE = 32

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

# Metadata constants
REQUIRED_METADATA_FIELDS = {'source', 'chunk_index'}
OPTIONAL_METADATA_FIELDS = {
    'document_id',
    'timestamp',
    'title',
    'author',
    'page_number',
    'section',
    'language',
    'confidence_score'
}

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
                        # Validate metadata before yielding
                        if 'metadata' not in current_chunk:
                            logger.warning(f"Missing metadata for chunk with text: {current_chunk['text'][:100]}...")
                            current_chunk['metadata'] = {'source': 'unknown', 'chunk_index': 0}
                        yield current_chunk
                    current_chunk = {"text": value}
                elif prefix.endswith('.metadata'):
                    if current_chunk is not None:
                        # Validate metadata structure
                        if not isinstance(value, dict):
                            logger.warning(f"Invalid metadata type: {type(value)}. Expected dict.")
                            value = {'source': 'unknown', 'chunk_index': 0}
                        current_chunk["metadata"] = value
        if current_chunk is not None:
            # Validate final chunk
            if 'metadata' not in current_chunk:
                logger.warning(f"Missing metadata for final chunk with text: {current_chunk['text'][:100]}...")
                current_chunk['metadata'] = {'source': 'unknown', 'chunk_index': 0}
            yield current_chunk
    except Exception as e:
        logger.error(f"Error in chunk generator: {str(e)}")
        logger.error(traceback.format_exc())
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
                        # Validate metadata before adding to batch
                        if 'metadata' not in current_chunk:
                            logger.warning(f"Missing metadata for chunk with text: {current_chunk['text'][:100]}...")
                            current_chunk['metadata'] = {'source': 'unknown', 'chunk_index': 0}
                        current_batch.append(current_chunk)
                        if len(current_batch) >= batch_size:
                            yield current_batch
                            current_batch = []
                            clear_memory()
                    current_chunk = {"text": value}
                elif prefix.endswith('.metadata'):
                    if current_chunk is not None:
                        # Validate metadata structure
                        if not isinstance(value, dict):
                            logger.warning(f"Invalid metadata type: {type(value)}. Expected dict.")
                            value = {'source': 'unknown', 'chunk_index': 0}
                        current_chunk["metadata"] = value
            
            # Add the last chunk if it exists
            if current_chunk is not None:
                # Validate final chunk metadata
                if 'metadata' not in current_chunk:
                    logger.warning(f"Missing metadata for final chunk with text: {current_chunk['text'][:100]}...")
                    current_chunk['metadata'] = {'source': 'unknown', 'chunk_index': 0}
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
        ValueError: If texts and metadata don't match or are invalid
    """
    if len(texts) != len(metadata):
        error_msg = f"Group {group_id}: Mismatch between texts ({len(texts)}) and metadata ({len(metadata)})"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    # Validate metadata structure and content
    valid_metadata = []
    for i, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            logging.warning(f"Group {group_id}: Invalid metadata at index {i}: expected dict, got {type(meta)}")
            meta = {'source': 'unknown', 'chunk_index': i}
        elif not meta:  # Empty dict
            logging.warning(f"Group {group_id}: Empty metadata at index {i}")
            meta = {'source': 'unknown', 'chunk_index': i}
        valid_metadata.append(meta)
    
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
        filtered_metadata = []
        for i in valid_indices:
            meta = valid_metadata[i]
            # Ensure metadata has required fields
            if 'source' not in meta:
                meta['source'] = 'unknown'
            if 'chunk_index' not in meta:
                meta['chunk_index'] = i
            filtered_metadata.append(meta)
        
        if len(embeddings) != len(filtered_metadata):
            error_msg = f"Group {group_id}: Mismatch between embeddings ({len(embeddings)}) and metadata ({len(filtered_metadata)})"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
        logging.info(f"Group {group_id}: Successfully processed {len(embeddings)} texts with metadata")
        return embeddings, filtered_metadata
        
    except Exception as e:
        logging.error(f"Group {group_id}: Processing failed - {str(e)}")
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Failed to process group {group_id}: {str(e)}")

def validate_metadata(metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
    """Validate and enrich metadata with default values for missing fields.
    
    Args:
        metadata: Original metadata dictionary
        chunk_index: Index of the chunk in the document
        
    Returns:
        Enriched metadata dictionary with all required and optional fields
    """
    if not isinstance(metadata, dict):
        metadata = {}
        
    # Ensure required fields
    enriched = {
        'source': metadata.get('source', 'unknown'),
        'chunk_index': chunk_index,
        'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
        'document_id': metadata.get('document_id', f"doc_{chunk_index}"),
        'title': metadata.get('title', ''),
        'author': metadata.get('author', ''),
        'page_number': metadata.get('page_number', 0),
        'section': metadata.get('section', ''),
        'language': metadata.get('language', 'en'),
        'confidence_score': metadata.get('confidence_score', 1.0)
    }
    
    # Validate field types
    if not isinstance(enriched['source'], str):
        enriched['source'] = str(enriched['source'])
    if not isinstance(enriched['chunk_index'], int):
        enriched['chunk_index'] = int(enriched['chunk_index'])
    if not isinstance(enriched['document_id'], str):
        enriched['document_id'] = str(enriched['document_id'])
    if not isinstance(enriched['title'], str):
        enriched['title'] = str(enriched['title'])
    if not isinstance(enriched['author'], str):
        enriched['author'] = str(enriched['author'])
    if not isinstance(enriched['page_number'], (int, float)):
        enriched['page_number'] = 0
    if not isinstance(enriched['section'], str):
        enriched['section'] = str(enriched['section'])
    if not isinstance(enriched['language'], str):
        enriched['language'] = 'en'
    if not isinstance(enriched['confidence_score'], (int, float)):
        enriched['confidence_score'] = 1.0
        
    return enriched

def get_metadata_stats(processed_chunks):
    """Generate statistics about the processed chunks' metadata.
    
    Args:
        processed_chunks (list): List of processed chunk dictionaries
        
    Returns:
        dict: Statistics about the metadata fields
    """
    stats = {
        'total_chunks': len(processed_chunks),
        'field_counts': {},
        'field_values': {},
        'missing_fields': {},
        'validation_errors': {}
    }
    
    for chunk in processed_chunks:
        metadata = chunk.get('metadata', {})
        
        # Count occurrences of each field
        for field in metadata:
            stats['field_counts'][field] = stats['field_counts'].get(field, 0) + 1
            
            # Track unique values for each field
            if field not in stats['field_values']:
                stats['field_values'][field] = set()
            stats['field_values'][field].add(str(metadata[field]))
            
        # Track missing required fields
        required_fields = {'source', 'chunk_index', 'timestamp'}
        missing = required_fields - set(metadata.keys())
        for field in missing:
            stats['missing_fields'][field] = stats['missing_fields'].get(field, 0) + 1
            
        # Track validation errors
        if 'validation_errors' in chunk:
            for error in chunk['validation_errors']:
                error_type = error.get('type', 'unknown')
                stats['validation_errors'][error_type] = stats['validation_errors'].get(error_type, 0) + 1
                
    # Convert sets to lists for JSON serialization
    for field in stats['field_values']:
        stats['field_values'][field] = list(stats['field_values'][field])
        
    return stats

def backup_metadata(metadata_list: List[Dict[str, Any]], backup_path: Path) -> bool:
    """Create a backup of metadata with timestamp.
    
    Args:
        metadata_list: List of metadata dictionaries
        backup_path: Path to save the backup
        
    Returns:
        True if backup was successful, False otherwise
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"metadata_backup_{timestamp}.json"
        
        with open(backup_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'metadata': metadata_list,
                'stats': get_metadata_stats(metadata_list)
            }, f, indent=2)
            
        logger.info(f"Created metadata backup at {backup_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create metadata backup: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def restore_metadata(backup_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Restore metadata from the most recent backup.
    
    Args:
        backup_path: Path containing metadata backups
        
    Returns:
        List of metadata dictionaries if successful, None otherwise
    """
    try:
        # Find most recent backup
        backup_files = list(backup_path.glob("metadata_backup_*.json"))
        if not backup_files:
            logger.error("No metadata backups found")
            return None
            
        latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_backup) as f:
            backup_data = json.load(f)
            
        logger.info(f"Restored metadata from backup {latest_backup}")
        return backup_data['metadata']
        
    except Exception as e:
        logger.error(f"Failed to restore metadata: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_chunk_group(group_id: int, chunks: List[Dict[str, Any]], rag: RAGSystem, 
                       output_dir: Path, backup_dir: Path) -> Tuple[int, int]:
    """Process a group of chunks in parallel.
    
    Args:
        group_id: ID of the group being processed
        chunks: List of chunk dictionaries
        rag: RAGSystem instance
        output_dir: Directory to save output files
        backup_dir: Directory to save metadata backups
        
    Returns:
        Tuple of (successful_chunks, failed_chunks)
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_chunk, chunk, rag, output_dir): chunk
                for chunk in chunks
            }
            
            # Collect results
            successful = 0
            failed = 0
            processed_chunks = []
            
            # Wait for all futures to complete
            done, _ = wait(futures.keys())
            
            # Process results
            for future in done:
                try:
                    result = future.result()
                    if result:
                        successful += 1
                        processed_chunks.append(result)
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    failed += 1
            
            # Backup metadata after successful processing
            if processed_chunks:
                backup_metadata(processed_chunks, backup_dir)
                
            return successful, failed
            
    except Exception as e:
        logger.error(f"Failed to process group {group_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return 0, len(chunks)

def process_chunk(chunk: Dict[str, Any], rag: RAGSystem, output_dir: Path) -> Optional[Dict[str, Any]]:
    """Process a single chunk.
    
    Args:
        chunk: Chunk dictionary containing text and metadata
        rag: RAGSystem instance
        output_dir: Directory to save output files
        
    Returns:
        Processed chunk metadata if successful, None otherwise
    """
    try:
        # Extract text and metadata
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        chunk_index = metadata.get('chunk_index', 0)
        
        if not text:
            logger.warning("Empty chunk text, skipping")
            return None
            
        # Validate and enrich metadata
        enriched_metadata = validate_metadata(metadata, chunk_index)
        
        # Add chunk to RAG system
        rag.add_text(text, enriched_metadata)
        
        # Save processed chunk info
        chunk_info = {
            'text': text[:100] + '...' if len(text) > 100 else text,  # Truncate for logging
            'metadata': enriched_metadata,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save to output file
        output_file = output_dir / f"chunk_{chunk_index}.json"
        with open(output_file, 'w') as f:
            json.dump(chunk_info, f, indent=2)
            
        return enriched_metadata
        
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to index chunks into the RAG system."""
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Create output and backup directories
        output_dir = Path("output/chunks")
        backup_dir = Path("output/metadata_backups")
        output_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load chunks from input file
        input_file = Path("input/chunks.json")
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return
            
        with open(input_file) as f:
            chunks = json.load(f)
            
        if not chunks:
            logger.error("No chunks found in input file")
            return
            
        # Group chunks for parallel processing
        chunk_groups = [chunks[i:i + GROUP_SIZE] for i in range(0, len(chunks), GROUP_SIZE)]
        
        # Process groups in parallel
        total_successful = 0
        total_failed = 0
        all_processed_chunks = []
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for group_id, group in enumerate(chunk_groups):
                future = executor.submit(process_chunk_group, group_id, group, rag, output_dir, backup_dir)
                futures.append(future)
                
            for future in futures:
                successful, failed = future.result()
                total_successful += successful
                total_failed += failed
                
        # Generate and save metadata statistics
        stats = get_metadata_stats(all_processed_chunks)
        stats_file = output_dir / "metadata_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Log summary
        logger.info(f"Processing complete:")
        logger.info(f"Total chunks processed: {len(chunks)}")
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"Metadata statistics saved to: {stats_file}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 