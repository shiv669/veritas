#!/usr/bin/env python3
"""
format_rag.py

Format input data into RAG chunks with:
 - Configurable chunk size
 - Metadata preservation
 - CLI support for input, output, and chunk size
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
from veritas.config import (
    INPUT_DATA_FILE,
    RAG_CHUNKS_FILE,
    DEFAULT_CHUNK_SIZE,
    ensure_directories,
    resolve_path,
    ensure_parent_dirs
)

# ─── CHUNKING FUNCTION ─────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int) -> list:
    """
    Split text into chunks of approximately chunk_size characters.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk
        
    Returns:
        List of text chunks
    """
    # Simple splitting by paragraphs first
    paragraphs = text.split("\n\n")
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk_size, save current chunk and start new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            # Otherwise add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ─── FORMAT FUNCTION ──────────────────────────────────────────────────────────
def format_rag(input_file: str, output_file: str, chunk_size: int) -> None:
    """
    Format input data into RAG chunks.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the RAG chunks
        chunk_size: Target size for each chunk
    """
    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process each document
    chunks = []
    for doc in data:
        doc_id = doc.get("id", "")
        title = doc.get("title", "")
        authors = doc.get("authors", [])
        year = doc.get("year", "")
        source = doc.get("source", "")
        text = doc.get("text", "")
        
        # Skip empty texts
        if not text:
            continue
        
        # Split text into chunks
        text_chunks = chunk_text(text, chunk_size)
        
        # Create chunk objects with metadata
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "id": f"{doc_id}_chunk_{i}",
                "title": title,
                "authors": authors,
                "year": year,
                "source": source,
                "text": chunk_text
            }
            chunks.append(chunk)
    
    # Save chunks
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(chunks)} chunks from {len(data)} documents.")

# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Format input data into RAG chunks."
    )
    parser.add_argument(
        "--input", type=str, default=str(INPUT_DATA_FILE),
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output", type=str, default=str(RAG_CHUNKS_FILE),
        help="Path to save the RAG chunks."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Target size for each chunk in characters."
    )
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()
    ensure_parent_dirs(resolve_path(args.output))

    # Format RAG chunks
    format_rag(args.input, args.output, args.chunk_size)

if __name__ == "__main__":
    main()
