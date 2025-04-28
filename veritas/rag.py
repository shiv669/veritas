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

from veritas.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAISS_TYPE,
    DEFAULT_NLIST,
    DEFAULT_BATCH_SIZE,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    RAG_CHUNKS_FILE
)
from veritas.utils import ensure_parent_dirs

class RAGSystem:
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        faiss_type: str = DEFAULT_FAISS_TYPE,
        nlist: int = DEFAULT_NLIST,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """Initialize the RAG system with specified parameters."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.faiss_type = faiss_type
        self.nlist = nlist
        self.batch_size = batch_size
        self.index = None
        self.metadata = []
        
    def process_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Process documents into chunks with metadata."""
        chunks = []
        for doc_idx, doc in enumerate(documents):
            # Simple chunking by splitting on whitespace
            words = doc.split()
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i:i + self.chunk_size])
                chunks.append({
                    "text": chunk,
                    "doc_id": doc_idx,
                    "chunk_id": len(chunks),
                    "start_idx": i,
                    "end_idx": min(i + self.chunk_size, len(words))
                })
        return chunks
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build FAISS index from document chunks."""
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        if self.faiss_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif self.faiss_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        self.metadata = chunks
        
        # Save index and metadata
        ensure_parent_dirs(FAISS_INDEX_FILE)
        ensure_parent_dirs(METADATA_FILE)
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f)
    
    def load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError("Index or metadata file not found")
        
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, 'r') as f:
            self.metadata = json.load(f)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return relevant chunks with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):  # Ensure index is valid
                chunk = self.metadata[idx].copy()
                chunk["score"] = float(distance)
                results.append(chunk)
        
        return results
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_name: str = "facebook/opt-350m",
        max_length: int = 200
    ) -> str:
        """Generate a response using retrieved chunks and language model."""
        # Prepare context from retrieved chunks
        context = "\n".join([chunk["text"] for chunk in retrieved_chunks])
        
        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Prepare prompt
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip() 