#!/usr/bin/env python3
"""
build_faiss_index.py

Builds a FAISS index for fast semantic search of document chunks.

What this file does:
This script creates a special search index that helps the AI quickly find 
relevant information in your documents. It's like creating a detailed table 
of contents or index for a book, but instead of just keywords, it can find 
content based on meaning and concepts.

After running this script, the AI will be able to instantly find relevant 
information to answer questions, even if you have thousands of documents.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from veritas.config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR,
    DEFAULT_CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL, ADVANCED_EMBEDDING_MODEL,
    DEFAULT_FAISS_TYPE, DEFAULT_NLIST,
    DEFAULT_TRAIN_SAMPLE, DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "build_faiss_index.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IndexConfig:
    """Configuration for FAISS index building."""
    index_type: str = DEFAULT_FAISS_TYPE
    nlist: int = DEFAULT_NLIST
    train_sample: int = DEFAULT_TRAIN_SAMPLE
    batch_size: int = DEFAULT_BATCH_SIZE
    workers: int = DEFAULT_WORKERS

@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = DEFAULT_BATCH_SIZE

class EmbeddingModel:
    """Wrapper for different embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = self._load_model()
        
    def _load_model(self) -> Union[SentenceTransformer, AutoModel]:
        """Load the specified embedding model."""
        try:
            if self.config.model_name in [DEFAULT_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL]:
                return SentenceTransformer(self.config.model_name, device=self.config.device)
            elif self.config.model_name == ADVANCED_EMBEDDING_MODEL:
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                model = AutoModel.from_pretrained(self.config.model_name).to(self.config.device)
                return (tokenizer, model)
            else:
                raise ValueError(f"Unsupported model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.config.model_name}: {e}")
            logger.info(f"Falling back to {FALLBACK_EMBEDDING_MODEL}")
            return SentenceTransformer(FALLBACK_EMBEDDING_MODEL, device=self.config.device)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the input texts."""
        try:
            if isinstance(self.model, SentenceTransformer):
                return self.model.encode(texts, batch_size=self.config.batch_size)
            else:
                tokenizer, model = self.model
                embeddings = []
                for i in range(0, len(texts), self.config.batch_size):
                    batch = texts[i:i + self.config.batch_size]
                    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
                return np.vstack(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

class FAISSIndexBuilder:
    """Builder for different types of FAISS indices."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        
    def build_index(self, dimension: int) -> faiss.Index:
        """Build a FAISS index of the specified type."""
        if self.config.index_type == "flat":
            return faiss.IndexFlatL2(dimension)
        elif self.config.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist, faiss.METRIC_L2)
            return index
        elif self.config.index_type == "hnsw":
            return faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
    
    def train_index(self, index: faiss.Index, embeddings: np.ndarray):
        """Train the index if needed."""
        if isinstance(index, faiss.IndexIVF):
            logger.info("Training IVF index...")
            index.train(embeddings[:self.config.train_sample])
        elif isinstance(index, faiss.IndexHNSW):
            logger.info("Training HNSW index...")
            index.train(embeddings[:self.config.train_sample])

def load_chunks(file_path: Path) -> List[Dict]:
    """Load text chunks from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chunks from {file_path}: {e}")
        raise

def build_index(
    chunks_file: Path,
    index_file: Path,
    metadata_file: Path,
    index_config: Optional[IndexConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None
) -> None:
    """
    Build FAISS index from text chunks.
    
    Args:
        chunks_file: Path to JSON file containing text chunks
        index_file: Path to save FAISS index
        metadata_file: Path to save metadata
        index_config: Configuration for FAISS index
        embedding_config: Configuration for embedding model
    """
    index_config = index_config or IndexConfig()
    embedding_config = embedding_config or EmbeddingConfig()
    
    logger.info("Loading chunks...")
    chunks = load_chunks(chunks_file)
    texts = [chunk["text"] for chunk in chunks]
    
    logger.info("Initializing embedding model...")
    embedding_model = EmbeddingModel(embedding_config)
    
    logger.info("Generating embeddings...")
    embeddings = []
    with ThreadPoolExecutor(max_workers=index_config.workers) as executor:
        futures = []
        for i in range(0, len(texts), index_config.batch_size):
            batch = texts[i:i + index_config.batch_size]
            futures.append(executor.submit(embedding_model.embed, batch))
        
        for future in tqdm(futures, desc="Generating embeddings"):
            embeddings.append(future.result())
    
    embeddings = np.vstack(embeddings)
    
    logger.info("Building index...")
    index_builder = FAISSIndexBuilder(index_config)
    index = index_builder.build_index(embeddings.shape[1])
    
    logger.info("Training index...")
    index_builder.train_index(index, embeddings)
    
    logger.info("Adding vectors to index...")
    index.add(embeddings)
    
    logger.info(f"Saving index to {index_file}...")
    faiss.write_index(index, str(index_file))
    
    logger.info(f"Saving metadata to {metadata_file}...")
    metadata = {
        "chunks": chunks,
        "index_config": vars(index_config),
        "embedding_config": vars(embedding_config)
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index from text chunks")
    parser.add_argument("--chunks-file", type=Path, default=DATA_DIR / "rag_chunks.json",
                      help="Path to JSON file containing text chunks")
    parser.add_argument("--index-file", type=Path, default=MODELS_DIR / "veritas_faiss.index",
                      help="Path to save FAISS index")
    parser.add_argument("--metadata-file", type=Path, default=MODELS_DIR / "veritas_metadata.pkl",
                      help="Path to save metadata")
    parser.add_argument("--index-type", type=str, default=DEFAULT_FAISS_TYPE,
                      help="Type of FAISS index to build (flat, ivf, hnsw)")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                      help="Name of the embedding model to use")
    
    args = parser.parse_args()
    
    index_config = IndexConfig(index_type=args.index_type)
    embedding_config = EmbeddingConfig(model_name=args.embedding_model)
    
    build_index(
        args.chunks_file,
        args.index_file,
        args.metadata_file,
        index_config,
        embedding_config
    )