#!/usr/bin/env python3
"""
Script to chunk JSON data into overlapping chunks for RAG processing.
"""

import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm

# Add the project root to Python path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.veritas.config import Config
from src.veritas.chunking import chunk_text

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    # Use the chunking implementation from the veritas package
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)

def process_json_file(input_file: str, output_dir: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Process JSON file and create chunks.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save chunked files
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    print(f"Loading {input_file}...")
    data = load_json_file(input_file)
    
    # Process each entry in the JSON
    chunked_data = []
    for idx, entry in enumerate(tqdm(data, desc="Processing entries")):
        # Convert entry to string for chunking
        entry_text = json.dumps(entry, ensure_ascii=False)
        
        # Create chunks
        chunks = create_chunks(entry_text, chunk_size, overlap)
        
        # Add metadata to each chunk
        for chunk_idx, chunk in enumerate(chunks):
            chunked_data.append({
                "chunk_id": f"{idx}_{chunk_idx}",
                "original_id": idx,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "content": chunk
            })
    
    # Save chunked data
    output_file = os.path.join(output_dir, "chunked_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(chunked_data)} chunks")
    print(f"Saved chunks to {output_file}")

if __name__ == "__main__":
    input_file = os.path.join(Config.INPUT_DIR, "1.json")
    output_dir = Config.CHUNKS_DIR
    chunk_size = Config.DEFAULT_CHUNK_SIZE
    overlap = Config.DEFAULT_CHUNK_OVERLAP
    
    process_json_file(input_file, output_dir, chunk_size, overlap) 