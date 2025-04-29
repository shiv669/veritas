#!/usr/bin/env python3
"""
improved_chunking.py

Improved chunking strategies for large documents in RAG systems.
This script provides enhanced chunking methods that are better suited for
academic papers and other large documents.
"""

import json
import re
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import nltk
from tqdm import tqdm

# Try to download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED = "fixed"  # Fixed-size chunks
    SENTENCE = "sentence"  # Sentence-based chunks
    PARAGRAPH = "paragraph"  # Paragraph-based chunks
    SEMANTIC = "semantic"  # Semantic-based chunks
    HYBRID = "hybrid"  # Hybrid approach combining multiple strategies
    HIERARCHICAL = "hierarchical"  # Hierarchical chunking for large documents

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    chunk_size: int = 512  # Target chunk size in tokens/words
    overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1000  # Maximum chunk size
    respect_sections: bool = True  # Whether to respect document sections
    section_markers: List[str] = None  # Markers for section boundaries
    max_chunks_per_doc: int = None  # Maximum number of chunks per document

class ImprovedChunker:
    """Enhanced chunker for different chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        if self.config.section_markers is None:
            self.config.section_markers = [
                r'^\d+\.\s+',  # Numbered sections (e.g., "1. Introduction")
                r'^[A-Z][a-z]+\s*:',  # Section headers (e.g., "Introduction:")
                r'^Abstract',  # Abstract section
                r'^Introduction',  # Introduction section
                r'^Methods?',  # Methods section
                r'^Results?',  # Results section
                r'^Discussion',  # Discussion section
                r'^Conclusion',  # Conclusion section
                r'^References?',  # References section
                r'^Appendix',  # Appendix section
                r'^\d+\.\d+\s+',  # Subsections (e.g., "1.1 Background")
            ]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return nltk.sent_tokenize(text)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # More robust paragraph splitting
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _identify_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Identify document sections based on markers."""
        if not self.config.respect_sections:
            return [(text, 0, len(text))]
        
        sections = []
        lines = text.split('\n')
        current_section = []
        current_section_start = 0
        
        for i, line in enumerate(lines):
            is_section_header = any(re.match(pattern, line) for pattern in self.config.section_markers)
            
            if is_section_header and current_section:
                # End current section and start a new one
                section_text = '\n'.join(current_section)
                sections.append((section_text, current_section_start, i))
                current_section = [line]
                current_section_start = i
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            section_text = '\n'.join(current_section)
            sections.append((section_text, current_section_start, len(lines)))
        
        return sections
    
    def _create_overlapping_chunks(self, items: List[str]) -> List[str]:
        """Create overlapping chunks from items."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in items:
            item_size = len(item.split())
            
            if current_size + item_size > self.config.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [item]
                current_size = item_size
            else:
                current_chunk.append(item)
                current_size += item_size
            
            # Add overlap
            if len(current_chunk) > 1:
                overlap = current_chunk[:-1]
                chunks.append(' '.join(overlap))
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_section(self, text: str) -> List[str]:
        """Chunk text by sections first, then by sentences or paragraphs within sections."""
        sections = self._identify_sections(text)
        all_chunks = []
        
        for section_text, start, end in sections:
            # Skip very short sections
            if len(section_text.split()) < self.config.min_chunk_size:
                continue
                
            # For longer sections, use sentence-based chunking
            if len(section_text.split()) > self.config.chunk_size * 2:
                sentences = self._split_into_sentences(section_text)
                section_chunks = self._create_overlapping_chunks(sentences)
                all_chunks.extend(section_chunks)
            else:
                # For shorter sections, keep them as a single chunk
                all_chunks.append(section_text)
        
        return all_chunks
    
    def _hybrid_chunking(self, text: str) -> List[str]:
        """Hybrid chunking approach that adapts to document structure."""
        # First, identify sections
        sections = self._identify_sections(text)
        all_chunks = []
        
        for section_text, start, end in sections:
            # Skip very short sections
            if len(section_text.split()) < self.config.min_chunk_size:
                continue
            
            # For very large sections, use hierarchical chunking
            if len(section_text.split()) > self.config.chunk_size * 3:
                # First split by paragraphs
                paragraphs = self._split_into_paragraphs(section_text)
                
                # Then for each paragraph, split by sentences if needed
                for para in paragraphs:
                    if len(para.split()) > self.config.chunk_size:
                        sentences = self._split_into_sentences(para)
                        para_chunks = self._create_overlapping_chunks(sentences)
                        all_chunks.extend(para_chunks)
                    else:
                        all_chunks.append(para)
            # For medium sections, use sentence-based chunking
            elif len(section_text.split()) > self.config.chunk_size:
                sentences = self._split_into_sentences(section_text)
                section_chunks = self._create_overlapping_chunks(sentences)
                all_chunks.extend(section_chunks)
            # For smaller sections, keep as is
            else:
                all_chunks.append(section_text)
        
        return all_chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text according to the configured strategy."""
        if self.config.strategy == ChunkingStrategy.FIXED:
            # Simple fixed-size chunking
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.config.chunk_size - self.config.overlap):
                chunk = ' '.join(words[i:i + self.config.chunk_size])
                if len(chunk.split()) >= self.config.min_chunk_size:
                    chunks.append(chunk)
            return chunks
        
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            # Sentence-based chunking
            sentences = self._split_into_sentences(text)
            return self._create_overlapping_chunks(sentences)
        
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH:
            # Paragraph-based chunking
            paragraphs = self._split_into_paragraphs(text)
            return self._create_overlapping_chunks(paragraphs)
        
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            # Semantic-based chunking (placeholder - can be improved with ML)
            # For now, using paragraph-based as a simple approximation
            paragraphs = self._split_into_paragraphs(text)
            return self._create_overlapping_chunks(paragraphs)
        
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            # Hybrid chunking approach
            return self._hybrid_chunking(text)
        
        elif self.config.strategy == ChunkingStrategy.HIERARCHICAL:
            # Hierarchical chunking for large documents
            chunks = self._chunk_by_section(text)
            
            # Apply max_chunks_per_doc limit if specified
            if self.config.max_chunks_per_doc is not None and len(chunks) > self.config.max_chunks_per_doc:
                logger.warning(f"Document has {len(chunks)} chunks, limiting to {self.config.max_chunks_per_doc}")
                
                # If we need to reduce chunks, use a more aggressive chunking approach
                if len(chunks) > self.config.max_chunks_per_doc * 2:
                    # For very large documents, use fixed-size chunking with larger chunks
                    words = text.split()
                    chunk_size = max(self.config.chunk_size, len(words) // self.config.max_chunks_per_doc)
                    chunks = []
                    for i in range(0, len(words), chunk_size - self.config.overlap):
                        chunk = ' '.join(words[i:i + chunk_size])
                        if len(chunk.split()) >= self.config.min_chunk_size:
                            chunks.append(chunk)
                else:
                    # For moderately large documents, just take the first max_chunks_per_doc chunks
                    chunks = chunks[:self.config.max_chunks_per_doc]
            
            return chunks
        
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.config.strategy}")

def process_document(
    file_path: Path,
    chunking_config: Optional[ChunkingConfig] = None
) -> List[Dict]:
    """
    Process a document and chunk it according to the specified strategy.
    
    Args:
        file_path: Path to the document
        chunking_config: Configuration for chunking
    
    Returns:
        List of chunks with metadata
    """
    # Initialize chunker
    chunking_config = chunking_config or ChunkingConfig()
    chunker = ImprovedChunker(chunking_config)
    
    # Read the document
    with open(file_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    
    # Extract text to chunk
    text = doc.get('fullText', '')
    if not text:
        # If no fullText, try to use abstract or other fields
        text = doc.get('abstract', '')
        if not text:
            return []  # No text to chunk
    
    # Create chunks
    chunks = chunker.chunk_text(text)
    
    # Add metadata to chunks
    result = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            'text': chunk,
            'source': str(file_path),
            'chunk_index': i,
            'total_chunks': len(chunks),
            'doc_id': doc.get('coreId', doc.get('doi', str(file_path)))
        }
        # Copy other metadata
        for key, value in doc.items():
            if key not in ['fullText', 'text', 'chunk_index', 'total_chunks', 'doc_id']:
                chunk_data[key] = value
        result.append(chunk_data)
    
    return result

def process_directory(
    input_dir: Path,
    output_file: Path,
    chunking_config: Optional[ChunkingConfig] = None
) -> None:
    """
    Process all documents in a directory and save chunks to a file.
    
    Args:
        input_dir: Directory containing input documents
        output_file: Path to save formatted chunks
        chunking_config: Configuration for chunking
    """
    # Get all JSON files
    files = list(input_dir.glob('*.json'))
    
    if not files:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    
    # Process all files
    all_chunks = []
    for file_path in tqdm(files, desc="Processing documents"):
        try:
            chunks = process_document(file_path, chunking_config)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Save chunks
    logger.info(f"Saving {len(all_chunks)} chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved chunking for RAG")
    parser.add_argument("--input-dir", type=Path, required=True,
                      help="Directory containing input documents")
    parser.add_argument("--output-file", type=Path,
                      default=Path("data/improved_rag_chunks.json"),
                      help="Path to save formatted chunks")
    parser.add_argument("--chunking-strategy", type=str,
                      choices=[s.value for s in ChunkingStrategy],
                      default=ChunkingStrategy.HYBRID.value,
                      help="Chunking strategy to use")
    parser.add_argument("--chunk-size", type=int,
                      default=512,
                      help="Size of chunks (for fixed strategy)")
    parser.add_argument("--overlap", type=int,
                      default=50,
                      help="Overlap between chunks")
    parser.add_argument("--respect-sections", action="store_true",
                      help="Whether to respect document sections")
    
    args = parser.parse_args()
    
    # Create chunking config
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy(args.chunking_strategy),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        respect_sections=args.respect_sections
    )
    
    # Process documents
    process_directory(
        args.input_dir,
        args.output_file,
        chunking_config
    ) 