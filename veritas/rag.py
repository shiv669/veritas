"""
RAG (Retrieval-Augmented Generation) implementation for Veritas.
This module provides the core functionality for document processing,
embedding generation, and retrieval operations.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pickle
import logging

from veritas.config import (
    # Model settings
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEN_MODEL,
    MAX_SEQ_LENGTH,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    
    # Performance settings
    EMBED_BATCH_SIZE,
    GEN_BATCH_SIZE,
    MAX_SEQ_LENGTH as EMBED_MAX_LENGTH,
    MAX_SEQ_LENGTH as GEN_MAX_LENGTH,
    
    # Indexing settings
    CHUNK_SIZE as DEFAULT_CHUNK_SIZE,
    DEFAULT_FAISS_TYPE,
    DEFAULT_NLIST,
    
    # File paths
    FAISS_INDEX_PATH as FAISS_INDEX_FILE,
    METADATA_PATH as METADATA_FILE,
    
    # Device settings
    DEVICE,
    get_device
)
from veritas.utils import ensure_parent_dirs
from .mps_utils import optimize_for_mps, prepare_inputs_for_mps
from .chunking import Chunker, ChunkingConfig, ChunkingStrategy

# Configure logging
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, device=None):
        """Initialize the RAG system."""
        # Set devices for different tasks
        self.inference_device = device or get_device()
        self.embedding_device = device or get_device()
        
        logger.info(f"Initializing RAG system with inference device: {self.inference_device}")
        logger.info(f"Using embedding device: {self.embedding_device}")
        
        # Initialize models on appropriate devices
        self.embedding_model = SentenceTransformer(
            DEFAULT_EMBEDDING_MODEL,
            device=self.embedding_device
        )
        
        self.generator = AutoModelForCausalLM.from_pretrained(
            DEFAULT_GEN_MODEL,
            device_map=None if self.inference_device in ["mps", "cpu"] else "auto",
            torch_dtype=torch.float32 if self.inference_device == "mps" else torch.float16
        )
        
        if self.inference_device in ["mps", "cpu"]:
            self.generator = self.generator.to(self.inference_device)
            if self.inference_device == "mps":
                self.generator = optimize_for_mps(self.generator)
        
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_GEN_MODEL)
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        
        # Initialize chunker with default configuration
        self.chunker = Chunker(ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=50,
            min_chunk_size=100,
            max_chunk_size=1000
        ))
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            device=self.embedding_device,
            max_length=EMBED_MAX_LENGTH
        )
    
    def generate_response(self, query, context):
        """Generate a response based on the query and context."""
        prompt = self._create_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Prepare inputs for the device
        inputs = prepare_inputs_for_mps(inputs, device=self.inference_device)
        
        # Generate response
        outputs = self.generator.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_SEQ_LENGTH,
            num_return_sequences=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Process documents and build FAISS index."""
        # Generate chunks using the chunker
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            # Extract content from document dictionary
            content = doc["content"] if isinstance(doc, dict) else doc
            chunks = self.chunker.chunk_text(content)
            # Add document index to chunk metadata
            for chunk in chunks:
                chunk["doc_id"] = doc_idx
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks were generated from the documents")
            return
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.generate_embeddings(texts)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        if DEFAULT_FAISS_TYPE == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Use Inner Product for normalized vectors
        elif DEFAULT_FAISS_TYPE == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, DEFAULT_NLIST, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        self.documents = all_chunks
        
        # Save index and metadata
        ensure_parent_dirs(FAISS_INDEX_FILE)
        ensure_parent_dirs(METADATA_FILE)
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors of dimension {dimension}")
        logger.info(f"Index saved to {FAISS_INDEX_FILE}")
        logger.info(f"Metadata saved to {METADATA_FILE}")
    
    def load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError("Index or metadata file not found")
        
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, 'rb') as f:
            self.documents = pickle.load(f)
    
    def retrieve(self, query: str, k: int = 5, min_score: float = 0.2) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])
        
        # Search in index
        scores, indices = self.index.search(query_embedding, k)
        
        # Filter and format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):  # Ensure index is valid
                chunk = self.documents[idx].copy()
                chunk["score"] = float(score)  # Convert numpy float to Python float
                results.append(chunk)
        
        # Sort by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Filter by minimum score
        results = [r for r in results if r["score"] >= min_score]
        
        return results
    
    def _create_prompt(self, query, context):
        """Create a prompt based on the query and context."""
        return f"""<s>[INST] You are a helpful AI assistant. Answer the question based on the following context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query} [/INST]""" 