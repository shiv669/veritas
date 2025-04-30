#!/usr/bin/env python3
"""
Script to chunk JSON data into overlapping chunks for RAG processing.
"""

import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm
import ijson

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
    Process JSON file by streaming entries and creating overlapping chunks in a memory-efficient manner.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "chunked_data.json")
    print(f"Streaming and chunking entries from {input_file} to {output_path}...")

    # Detect JSON format (array vs JSON Lines)
    is_json_array = False
    with open(input_file, 'r', encoding='utf-8') as detect_f:
        for ch in detect_f.read(1024):
            if ch.isspace():
                continue
            if ch == '[':
                is_json_array = True
            break

    # Open output and begin writing JSON array
    with open(output_path, 'w', encoding='utf-8') as out_f:
        # Start JSON array
        out_f.write('[\n')
        first = True

        # Stream through entries
        if is_json_array:
            in_stream = open(input_file, 'rb')
            entries = ijson.items(in_stream, 'item')
        else:
            # JSON Lines fallback
            def entries_gen():
                with open(input_file, 'r', encoding='utf-8') as text_f:
                    for line in text_f:
                        line = line.strip()
                        if not line:
                            continue
                        yield json.loads(line)
            entries = entries_gen()

        for idx, entry in enumerate(entries):
            # Convert entry to string for chunking
            entry_text = json.dumps(entry, ensure_ascii=False, default=str)
            # Create chunks
            chunks = create_chunks(entry_text, chunk_size, overlap)

            # Write each chunk as a JSON object
            for chunk_idx, text_chunk in enumerate(chunks):
                obj = {
                    'chunk_id': f"{idx}_{chunk_idx}",
                    'original_id': idx,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'content': text_chunk
                }
                # Prepend comma if not first
                if not first:
                    out_f.write(',\n')
                out_f.write(json.dumps(obj, ensure_ascii=False))
                first = False

        # End JSON array
        out_f.write('\n]\n')

    print(f"Created {idx+1} entries worth of chunks into {output_path}")

if __name__ == "__main__":
    # Use processed JSON file as input
    input_file = os.path.join(Config.DATA_DIR, "processed", "1.json")
    output_dir = Config.CHUNKS_DIR
    chunk_size = Config.DEFAULT_CHUNK_SIZE
    overlap = Config.DEFAULT_CHUNK_OVERLAP
    
    process_json_file(input_file, output_dir, chunk_size, overlap) 