#!/usr/bin/env python3
"""
chunking.py

Module for advanced document chunking strategies in the Veritas RAG system.
"""

import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED = "fixed"  # Fixed-size chunks
    SENTENCE = "sentence"  # Sentence-based chunks
    PARAGRAPH = "paragraph"  # Paragraph-based chunks
    SEMANTIC = "semantic"  # Semantic-based chunks
    HYBRID = "hybrid"  # Hybrid approach combining multiple strategies

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    chunk_size: int = 512  # Target chunk size in words
    overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1000  # Maximum chunk size
    respect_sections: bool = True  # Whether to respect document sections
    max_chunks_per_doc: Optional[int] = None  # Maximum chunks per document

class Chunker:
    """Advanced chunker with multiple chunking strategies."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Handle common abbreviations
        text = re.sub(r'(?<=Mr)\.(?=\s[A-Z])', '@POINT@', text)
        text = re.sub(r'(?<=Mrs)\.(?=\s[A-Z])', '@POINT@', text)
        text = re.sub(r'(?<=Dr)\.(?=\s[A-Z])', '@POINT@', text)
        text = re.sub(r'(?<=Prof)\.(?=\s[A-Z])', '@POINT@', text)
        text = re.sub(r'(?<=Sr)\.(?=\s[A-Z])', '@POINT@', text)
        text = re.sub(r'(?<=Jr)\.(?=\s[A-Z])', '@POINT@', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviation points
        sentences = [s.replace('@POINT@', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, handling various newline formats and headings."""
        # Normalize newlines
        text = text.replace('\r\n', '\n')
        
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up paragraphs and handle headings
        cleaned_paragraphs = []
        for p in paragraphs:
            # Clean up whitespace
            p = re.sub(r'\s+', ' ', p.strip())
            
            # Skip empty paragraphs
            if not p:
                continue
            
            # Handle bullet points and numbered lists
            if re.match(r'^[\d\-\*]\.\s+', p):
                # Split on bullet points or numbers
                items = re.split(r'(?<=\n)(?=[\d\-\*]\.\s+)', p)
                cleaned_paragraphs.extend(items)
            else:
                cleaned_paragraphs.append(p)
        
        return [p for p in cleaned_paragraphs if p]
    
    def _create_overlapping_chunks(self, items: List[str], size_fn=len) -> List[str]:
        """Create overlapping chunks from items."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in items:
            item_size = size_fn(item.split())
            
            if current_size + item_size > self.config.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Create overlap by keeping some items
                    overlap_size = 0
                    overlap_items = []
                    for prev_item in reversed(current_chunk):
                        prev_size = size_fn(prev_item.split())
                        if overlap_size + prev_size <= self.config.overlap:
                            overlap_items.insert(0, prev_item)
                            overlap_size += prev_size
                        else:
                            break
                    current_chunk = overlap_items + [item]
                    current_size = sum(size_fn(x.split()) for x in current_chunk)
                else:
                    current_chunk = [item]
                    current_size = item_size
            else:
                current_chunk.append(item)
                current_size += item_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_strategy(self, text: str, strategy: ChunkingStrategy) -> List[str]:
        """Chunk text using a specific strategy."""
        if strategy == ChunkingStrategy.FIXED:
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.config.chunk_size - self.config.overlap):
                chunk = ' '.join(words[i:i + self.config.chunk_size])
                if len(chunk.split()) >= self.config.min_chunk_size:
                    chunks.append(chunk)
            return chunks
        
        elif strategy == ChunkingStrategy.SENTENCE:
            sentences = self._split_into_sentences(text)
            return self._create_overlapping_chunks(sentences)
        
        elif strategy == ChunkingStrategy.PARAGRAPH:
            paragraphs = self._split_into_paragraphs(text)
            return self._create_overlapping_chunks(paragraphs)
        
        elif strategy == ChunkingStrategy.SEMANTIC:
            # Start with paragraph-based chunking
            paragraphs = self._split_into_paragraphs(text)
            
            # For paragraphs that are too long, split into sentences
            chunks = []
            for para in paragraphs:
                if len(para.split()) > self.config.max_chunk_size:
                    sentences = self._split_into_sentences(para)
                    chunks.extend(self._create_overlapping_chunks(sentences))
                else:
                    chunks.append(para)
            
            # Filter out chunks that are too small
            chunks = [chunk for chunk in chunks if len(chunk.split()) >= self.config.min_chunk_size]
            
            return chunks
        
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text according to the configured strategy.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of dictionaries containing chunks and their metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean up text
        text = text.strip()
        
        # Apply chunking strategy
        if self.config.strategy == ChunkingStrategy.HYBRID:
            # Try paragraph-based chunking first
            chunks = self._chunk_by_strategy(text, ChunkingStrategy.PARAGRAPH)
            
            # If chunks are too large, split them further
            final_chunks = []
            for chunk in chunks:
                if len(chunk.split()) > self.config.max_chunk_size:
                    # Split large chunks into sentences
                    sentences = self._split_into_sentences(chunk)
                    sentence_chunks = self._create_overlapping_chunks(sentences)
                    final_chunks.extend(sentence_chunks)
                else:
                    final_chunks.append(chunk)
            
            chunks = final_chunks
        else:
            chunks = self._chunk_by_strategy(text, self.config.strategy)
        
        # Apply max_chunks_per_doc limit if specified
        if self.config.max_chunks_per_doc and len(chunks) > self.config.max_chunks_per_doc:
            logger.warning(
                f"Document has {len(chunks)} chunks, limiting to {self.config.max_chunks_per_doc}"
            )
            chunks = chunks[:self.config.max_chunks_per_doc]
        
        # Create chunk metadata
        result = []
        for i, chunk in enumerate(chunks):
            # Skip chunks that are too small
            if len(chunk.split()) < self.config.min_chunk_size:
                continue
                
            result.append({
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "word_count": len(chunk.split()),
                "strategy": self.config.strategy.value
            })
        
        return result 