#!/usr/bin/env python3
"""
search_faiss.py

Search FAISS index with support for different search strategies and output formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import faiss
from dataclasses import dataclass
from enum import Enum
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from veritas.config import (
    MODELS_DIR, LOGS_DIR,
    DEFAULT_TOP_K, EMBED_PROMPT,
    DEFAULT_EMBEDDING_MODEL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "search_faiss.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """Available search strategies."""
    EXACT = "exact"  # Exact search with L2 distance
    APPROXIMATE = "approximate"  # Approximate search with IVF
    HNSW = "hnsw"  # HNSW search

class OutputFormat(Enum):
    """Available output formats."""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"

@dataclass
class SearchConfig:
    """Configuration for FAISS search."""
    strategy: SearchStrategy = SearchStrategy.EXACT
    top_k: int = DEFAULT_TOP_K
    nprobe: int = 10  # For IVF search
    ef_search: int = 64  # For HNSW search

@dataclass
class FilterConfig:
    """Configuration for result filtering."""
    min_score: float = 0.0
    max_score: float = float('inf')
    metadata_filters: Dict[str, Any] = None

class FAISSSearcher:
    """Searcher for FAISS index with different strategies."""
    
    def __init__(self, index: faiss.Index, metadata: Dict):
        self.index = index
        self.metadata = metadata
        self.config = SearchConfig()
    
    def set_config(self, config: SearchConfig):
        """Set search configuration."""
        self.config = config
        
        # Configure index parameters based on strategy
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = config.nprobe
        elif isinstance(self.index, faiss.IndexHNSW):
            self.index.setEfSearch(config.ef_search)
    
    def search(self, query_vector: np.ndarray) -> tuple:
        """Search the index with the configured strategy."""
        if self.config.strategy == SearchStrategy.EXACT:
            return self.index.search(query_vector, self.config.top_k)
        elif self.config.strategy == SearchStrategy.APPROXIMATE:
            if not isinstance(self.index, faiss.IndexIVF):
                raise ValueError("Approximate search requires an IVF index")
            return self.index.search(query_vector, self.config.top_k)
        elif self.config.strategy == SearchStrategy.HNSW:
            if not isinstance(self.index, faiss.IndexHNSW):
                raise ValueError("HNSW search requires an HNSW index")
            return self.index.search(query_vector, self.config.top_k)
        else:
            raise ValueError(f"Unsupported search strategy: {self.config.strategy}")

class ResultFormatter:
    """Formatter for search results in different formats."""
    
    @staticmethod
    def format_json(results: List[Dict]) -> str:
        """Format results as JSON."""
        return json.dumps(results, indent=2)
    
    @staticmethod
    def format_text(results: List[Dict]) -> str:
        """Format results as plain text."""
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"Result {i} (Score: {result['score']:.4f}):")
            output.append(f"Text: {result['text']}")
            if result.get('metadata'):
                output.append("Metadata:")
                for key, value in result['metadata'].items():
                    output.append(f"  {key}: {value}")
            output.append("")
        return "\n".join(output)
    
    @staticmethod
    def format_markdown(results: List[Dict]) -> str:
        """Format results as markdown."""
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"## Result {i} (Score: {result['score']:.4f})")
            output.append("")
            output.append(result['text'])
            if result.get('metadata'):
                output.append("")
                output.append("### Metadata")
                for key, value in result['metadata'].items():
                    output.append(f"- **{key}**: {value}")
            output.append("")
        return "\n".join(output)

def filter_results(
    results: List[Dict],
    filter_config: FilterConfig
) -> List[Dict]:
    """Filter search results based on configuration."""
    filtered = []
    for result in results:
        # Score filtering
        if not (filter_config.min_score <= result['score'] <= filter_config.max_score):
            continue
        
        # Metadata filtering
        if filter_config.metadata_filters:
            metadata = result.get('metadata', {})
            matches = True
            for key, value in filter_config.metadata_filters.items():
                if key not in metadata or metadata[key] != value:
                    matches = False
                    break
            if not matches:
                continue
        
        filtered.append(result)
    
    return filtered

def search_index(
    query: str,
    index_file: Path,
    metadata_file: Path,
    embedding_model: Any,
    search_config: Optional[SearchConfig] = None,
    filter_config: Optional[FilterConfig] = None,
    output_format: OutputFormat = OutputFormat.JSON
) -> str:
    """
    Search FAISS index with the given query.
    
    Args:
        query: Search query
        index_file: Path to FAISS index
        metadata_file: Path to metadata file
        embedding_model: Embedding model instance
        search_config: Search configuration
        filter_config: Filter configuration
        output_format: Output format
    
    Returns:
        Formatted search results
    """
    # Load index and metadata
    logger.info(f"Loading index from {index_file}...")
    index = faiss.read_index(str(index_file))
    
    logger.info(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Initialize searcher
    searcher = FAISSSearcher(index, metadata)
    if search_config:
        searcher.set_config(search_config)
    
    # Generate query embedding
    logger.info("Generating query embedding...")
    query_vector = embedding_model.embed([query])[0].reshape(1, -1)
    
    # Search
    logger.info("Searching index...")
    distances, indices = searcher.search(query_vector)
    
    # Format results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0:  # Invalid index
            continue
        
        chunk = metadata['chunks'][idx]
        result = {
            'score': float(distance),
            'text': chunk['text'],
            'metadata': {
                k: v for k, v in chunk.items()
                if k != 'text'
            }
        }
        results.append(result)
    
    # Filter results
    if filter_config:
        results = filter_results(results, filter_config)
    
    # Format output
    if output_format == OutputFormat.JSON:
        return ResultFormatter.format_json(results)
    elif output_format == OutputFormat.TEXT:
        return ResultFormatter.format_text(results)
    elif output_format == OutputFormat.MARKDOWN:
        return ResultFormatter.format_markdown(results)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

if __name__ == "__main__":
    import argparse
    from sentence_transformers import SentenceTransformer
    
    parser = argparse.ArgumentParser(description="Search FAISS index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--index-file", type=Path, default=MODELS_DIR / "veritas_faiss.index",
                      help="Path to FAISS index")
    parser.add_argument("--metadata-file", type=Path, default=MODELS_DIR / "veritas_metadata.pkl",
                      help="Path to metadata file")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                      help="Name of the embedding model to use")
    parser.add_argument("--strategy", type=str, choices=[s.value for s in SearchStrategy],
                      default=SearchStrategy.EXACT.value, help="Search strategy")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                      help="Number of results to return")
    parser.add_argument("--output-format", type=str, choices=[f.value for f in OutputFormat],
                      default=OutputFormat.JSON.value, help="Output format")
    
    args = parser.parse_args()

    # Load embedding model
    embedding_model = SentenceTransformer(args.embedding_model)

    # Create configurations
    search_config = SearchConfig(
        strategy=SearchStrategy(args.strategy),
        top_k=args.top_k
    )

    # Search and print results
    results = search_index(
        args.query,
        args.index_file,
        args.metadata_file,
        embedding_model,
        search_config,
        output_format=OutputFormat(args.output_format)
    )
    print(results)
