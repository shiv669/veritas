"""
Text chunking utilities for the Veritas RAG system
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
    Calculate an appropriate chunk size based on text length and target number of chunks
    
    Args:
        text_length: Length of the text in characters
        target_chunks: Target number of chunks
        
    Returns:
        Recommended chunk size
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
    Split text into chunks of specified size with overlap
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
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