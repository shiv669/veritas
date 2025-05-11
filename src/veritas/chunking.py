"""
Text chunking utilities for the Veritas RAG system

This module provides functionality for dividing large documents into smaller,
semantically meaningful chunks that can be processed efficiently by the RAG system.
Effective chunking is crucial for retrieval accuracy and model performance.

Features:
- Automatic chunk size determination
- Intelligent text segmentation
- Sentence-aware chunking
- Overlap control for context preservation

Usage examples:
    from veritas.chunking import chunk_text, get_chunk_size
    
    # Determine optimal chunk size for a document
    optimal_size = get_chunk_size(len(document_text))
    
    # Split document into chunks with default settings
    chunks = chunk_text(document_text)
    
    # Split with custom parameters
    chunks = chunk_text(document_text, chunk_size=500, overlap=100)
"""
import re
import nltk
from typing import List, Tuple, Dict, Any
from .config import Config

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def get_chunk_size(text_length: int, target_chunks: int = 10) -> int:
    """
    Calculate optimal chunk size based on document length and target number of chunks.
    
    This function dynamically determines an appropriate chunk size by dividing
    the total text length by the desired number of chunks, with constraints to
    ensure chunks are neither too small nor too large.
    
    Args:
        text_length: Total length of the document in characters
        target_chunks: Desired number of chunks to produce
        
    Returns:
        int: Recommended chunk size in characters
        
    Example:
        document_length = len(large_document)
        chunk_size = get_chunk_size(document_length, target_chunks=15)
        chunks = chunk_text(large_document, chunk_size=chunk_size)
    """
    if text_length <= 0:
        return Config.DEFAULT_CHUNK_SIZE
    
    chunk_size = text_length // target_chunks
    
    # Ensure chunk size is reasonable
    if chunk_size < 100:
        return 100
    elif chunk_size > 2000:
        return 2000
    
    return chunk_size


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into semantically meaningful chunks with controlled overlap.
    
    This function segments text into chunks by respecting sentence boundaries
    where possible, and handles special cases like very long sentences by
    breaking them at punctuation or word boundaries. The overlap parameter
    helps maintain context between adjacent chunks.
    
    Args:
        text: The text content to be chunked
        chunk_size: Maximum size of each chunk in characters (default from Config)
        overlap: Number of characters to overlap between chunks (default from Config)
        
    Returns:
        List[str]: List of text chunks ready for embedding and indexing
        
    Example:
        document = "This is a long document that needs to be split..."
        chunks = chunk_text(document, chunk_size=500, overlap=50)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk[:50]}...")
    """
    if not text or not text.strip():
        return []
    
    if chunk_size is None:
        chunk_size = Config.DEFAULT_CHUNK_SIZE
    
    if overlap is None:
        overlap = Config.DEFAULT_CHUNK_OVERLAP
    
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If a single sentence is longer than chunk_size, split it
        if sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            # Split long sentence by punctuation if possible
            parts = re.split(r'[,;:]\s+', sentence)
            for part in parts:
                part_length = len(part)
                if part_length > chunk_size:
                    # Further split by words if still too long
                    words = part.split()
                    part_chunk = []
                    part_length = 0
                    
                    for word in words:
                        if part_length + len(word) + 1 > chunk_size:
                            chunks.append(" ".join(part_chunk))
                            part_chunk = [word]
                            part_length = len(word)
                        else:
                            part_chunk.append(word)
                            part_length += len(word) + 1
                            
                    if part_chunk:
                        chunks.append(" ".join(part_chunk))
                else:
                    current_chunk.append(part)
                    current_length += part_length + 1
                    
                    if current_length >= chunk_size:
                        chunks.append(" ".join(current_chunk))
                        
                        # Calculate overlap
                        if overlap > 0:
                            overlap_text = " ".join(current_chunk[-overlap:])
                            current_chunk = current_chunk[-overlap:]
                            current_length = len(overlap_text)
                        else:
                            current_chunk = []
                            current_length = 0
        else:
            # Add regular sentence to current chunk
            if current_length + sentence_length + 1 > chunk_size:
                chunks.append(" ".join(current_chunk))
                
                # Calculate overlap
                if overlap > 0:
                    overlap_words = " ".join(current_chunk[-overlap:])
                    current_chunk = current_chunk[-overlap:]
                    current_length = len(overlap_words)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks 