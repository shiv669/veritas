#!/usr/bin/env python3
"""
format_rag.py

Format documents for RAG with support for different input formats and chunking strategies.

What this file does:
This script prepares your documents so they can be searched by the AI.
It's like an automated librarian that:
1. Takes your documents (PDFs, text files, etc.)
2. Breaks them into manageable pieces
3. Organizes them so the AI can quickly find relevant information

You run this before using Veritas to get your documents ready for searching.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import sys
import re
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from veritas.config import (
    DATA_DIR, LOGS_DIR,
    DEFAULT_CHUNK_SIZE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "format_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InputFormat(Enum):
    """
    Supported file types that Veritas can process
    
    This tells the system what kind of documents you're feeding it.
    """
    JSON = "json"
    TXT = "txt"
    PDF = "pdf"
    MARKDOWN = "md"
    HTML = "html"

class ChunkingStrategy(Enum):
    """
    Different ways to split your documents into chunks
    
    Think of these like different ways to divide a book:
    - FIXED: Split every X words
    - SENTENCE: Split at the end of sentences
    - PARAGRAPH: Split at paragraph breaks
    - SEMANTIC: Try to keep related concepts together
    """
    FIXED = "fixed"  # Fixed-size chunks
    SENTENCE = "sentence"  # Sentence-based chunks
    PARAGRAPH = "paragraph"  # Paragraph-based chunks
    SEMANTIC = "semantic"  # Semantic-based chunks

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1000  # Maximum chunk size

class DocumentProcessor:
    """Processor for different document formats."""
    
    @staticmethod
    def process_json(file_path: Path) -> List[Dict]:
        """Process JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")
    
    @staticmethod
    def process_txt(file_path: Path) -> List[Dict]:
        """Process text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [{"text": text, "source": str(file_path)}]
    
    @staticmethod
    def process_markdown(file_path: Path) -> List[Dict]:
        """Process markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [{"text": text, "source": str(file_path)}]
    
    @staticmethod
    def process_html(file_path: Path) -> List[Dict]:
        """Process HTML file."""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return [{"text": text, "source": str(file_path)}]
        except ImportError:
            logger.error("BeautifulSoup4 is required for HTML processing")
            raise
    
    @staticmethod
    def process_pdf(file_path: Path) -> List[Dict]:
        """Process PDF file."""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return [{"text": text, "source": str(file_path)}]
        except ImportError:
            logger.error("PyPDF2 is required for PDF processing")
            raise

class TextChunker:
    """Chunker for different chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Basic sentence splitting - can be improved with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
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
        
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.config.strategy}")

def process_document(
    file_path: Path,
    input_format: InputFormat,
    chunking_config: Optional[ChunkingConfig] = None
) -> List[Dict]:
    """
    Process a document and chunk it according to the specified strategy.
    
    Args:
        file_path: Path to the document
        input_format: Format of the input document
        chunking_config: Configuration for chunking
    
    Returns:
        List of chunks with metadata
    """
    # Initialize chunker
    chunking_config = chunking_config or ChunkingConfig()
    chunker = TextChunker(chunking_config)
    
    # Process document based on format
    if input_format == InputFormat.JSON:
        documents = DocumentProcessor.process_json(file_path)
    elif input_format == InputFormat.TXT:
        documents = DocumentProcessor.process_txt(file_path)
    elif input_format == InputFormat.PDF:
        documents = DocumentProcessor.process_pdf(file_path)
    elif input_format == InputFormat.MARKDOWN:
        documents = DocumentProcessor.process_markdown(file_path)
    elif input_format == InputFormat.HTML:
        documents = DocumentProcessor.process_html(file_path)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    
    # Chunk documents
    chunks = []
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Create chunks
        doc_chunks = chunker.chunk_text(text)
        
        # Add metadata to chunks
        for i, chunk in enumerate(doc_chunks):
            chunk_data = {
                'text': chunk,
                'source': doc.get('source', str(file_path)),
                'chunk_index': i,
                'total_chunks': len(doc_chunks)
            }
            # Copy other metadata
            for key, value in doc.items():
                if key != 'text':
                    chunk_data[key] = value
            chunks.append(chunk_data)
    
    return chunks

def format_documents(
    input_dir: Path,
    output_file: Path,
    input_format: InputFormat,
    chunking_config: Optional[ChunkingConfig] = None
) -> None:
    """
    Format all documents in a directory for RAG.
    
    Args:
        input_dir: Directory containing input documents
        output_file: Path to save formatted chunks
        input_format: Format of input documents
        chunking_config: Configuration for chunking
    """
    # Get all files with the specified format
    pattern = f"*.{input_format.value}"
    files = list(input_dir.glob(pattern))
    
    if not files:
        logger.warning(f"No {input_format.value} files found in {input_dir}")
        return
    
    # Process all files
    all_chunks = []
    for file_path in tqdm(files, desc="Processing documents"):
        try:
            chunks = process_document(file_path, input_format, chunking_config)
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
    
    parser = argparse.ArgumentParser(description="Format documents for RAG")
    parser.add_argument("--input-dir", type=Path, required=True,
                      help="Directory containing input documents")
    parser.add_argument("--output-file", type=Path,
                      default=DATA_DIR / "rag_chunks.json",
                      help="Path to save formatted chunks")
    parser.add_argument("--input-format", type=str,
                      choices=[f.value for f in InputFormat],
                      required=True, help="Format of input documents")
    parser.add_argument("--chunking-strategy", type=str,
                      choices=[s.value for s in ChunkingStrategy],
                      default=ChunkingStrategy.FIXED.value,
                      help="Chunking strategy to use")
    parser.add_argument("--chunk-size", type=int,
                      default=DEFAULT_CHUNK_SIZE,
                      help="Size of chunks (for fixed strategy)")
    parser.add_argument("--overlap", type=int,
                      default=50,
                      help="Overlap between chunks")
    
    args = parser.parse_args()
    
    # Create chunking config
    chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy(args.chunking_strategy),
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Format documents
    format_documents(
        args.input_dir,
        args.output_file,
        InputFormat(args.input_format),
        chunking_config
    )
