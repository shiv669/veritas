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
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        faiss_type: str = "flat",
        nlist: int = 100,
        batch_size: int = 32
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
            batch_embeddings = self.embedding_model.encode(batch, normalize_embeddings=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        if self.faiss_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Use Inner Product for normalized vectors
        elif self.faiss_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        self.metadata = chunks
        
        # Save index and metadata
        ensure_parent_dirs(FAISS_INDEX_FILE)
        ensure_parent_dirs(METADATA_FILE)
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError("Index or metadata file not found")
        
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, 'rb') as f:
            self.metadata = pickle.load(f)
    
    def retrieve(self, query: str, k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a query.
        
        Args:
            query: The query string to search for
            k: Number of results to return
            min_score: Minimum similarity score threshold (0 to 1)
            
        Returns:
            List of dictionaries containing the retrieved chunks and their metadata
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in index
        scores, indices = self.index.search(query_embedding, k * 2)  # Get more results to filter
        
        # Filter and deduplicate results
        seen_contents = set()
        results = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):  # Ensure index is valid
                chunk = self.metadata[idx].copy()
                
                # Convert distance to similarity score (for inner product, higher is better)
                score = float(score)
                
                # Get the content to check for duplicates
                content = str(chunk.get('content', chunk.get('text', '')))
                
                # Skip if score is too low or content is duplicate
                if score < min_score or content in seen_contents:
                    continue
                
                chunk["score"] = score
                results.append(chunk)
                seen_contents.add(content)
                
                # Break if we have enough results
                if len(results) >= k:
                    break
        
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