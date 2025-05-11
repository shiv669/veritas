"""
RAG (Retrieval-Augmented Generation) implementation for Veritas

What this file does:
This is the heart of Veritas - it lets the AI answer questions by:
1. Finding relevant information in your documents (Retrieval)
2. Using that information to generate accurate answers (Generation)

Think of it like giving the AI the ability to "look things up" before answering!
"""
import os
import torch
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from .config import Config, get_device
from .utils import setup_logging

logger = setup_logging(__name__)

class RAGSystem:
    """
    The main RAG system that combines retrieval and generation
    
    What it does:
    - Finds relevant chunks of text from your documents
    - Feeds those chunks to the AI to generate accurate answers
    - Keeps track of where information came from for citations
    """
    
    def __init__(self, 
                 embedding_model: str = None,
                 llm_model: str = None,
                 index_path: str = None,
                 device: str = None):
        """
        Sets up the RAG system with all needed components
        
        Parameters:
        - embedding_model: Model that converts text to numbers (vectors)
        - llm_model: The AI model that generates answers (Mistral 2 7B)
        - index_path: Where your document index is stored
        - device: What hardware to use (GPU, Apple Silicon, or CPU)
        """
        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load embedding model
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        
        # Load LLM
        self.llm_model_name = llm_model or Config.LLM_MODEL
        logger.info(f"Loading language model: {self.llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_model_name, torch_dtype=torch.float16)
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        elif self.device == "mps":
            self.model = self.model.to("mps")
            
        # Setup generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        # Load FAISS index if provided
        self.index = None
        self.chunks = []
        if index_path:
            self.load_index(index_path)
    
    def load_index(self, index_path: str) -> None:
        """
        Load FAISS index and chunks
        
        Args:
            index_path: Path to FAISS index directory
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path not found: {index_path}")
            
        # Load chunks (npy or fallback to JSON)
        chunks_npy = os.path.join(os.path.dirname(index_path), "chunks.npy")
        chunks_json = os.path.join(os.path.dirname(index_path), "chunks.json")
        if os.path.exists(chunks_npy):
            self.chunks = np.load(chunks_npy, allow_pickle=True).tolist()
        elif os.path.exists(chunks_json):
            import json
            with open(chunks_json, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        else:
            logger.warning(f"Chunks file not found (expected chunks.npy or chunks.json) in: {index_path}")
        
        # Load index
        index_file = os.path.join(os.path.dirname(index_path), "index.faiss")
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            logger.warning(f"Index file not found: {index_file}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query using the embedding model
        
        Args:
            query: Query to embed
            
        Returns:
            Query embedding
        """
        return self.embedding_model.encode(query, normalize_embeddings=True)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for the query
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        if not self.index or not self.chunks:
            logger.error("Index or chunks not loaded")
            return []
        
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Search index
        scores, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), top_k
        )
        
        # Construct results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Handle different chunk formats
                if isinstance(chunk, dict):
                    text = chunk.get('text', str(chunk))
                else:
                    text = str(chunk)
                
                results.append({
                    "score": float(score),
                    "chunk": text,
                    "index": int(idx)
                })
        
        return results
    
    def generate(self, prompt: str, 
                 max_new_tokens: int = 512, 
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """
        Generate text using the language model
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1
        )
        
        return result[0]["generated_text"][len(prompt):]


def query_rag(query: str, rag_system: RAGSystem = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Process a query through the RAG system
    
    Args:
        query: The user's query
        rag_system: RAG system instance or None to create a new one
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with query results
    """
    # Create RAG system if not provided
    if rag_system is None:
        logger.info("Creating new RAG system")
        index_path = os.path.join(Config.MODELS_DIR, "faiss", "index.faiss")
        rag_system = RAGSystem(index_path=index_path)
    
    # Retrieve relevant chunks
    retrieved_chunks = rag_system.retrieve(query, top_k=top_k)
    
    # Construct prompt
    context = "\n\n".join([item["chunk"] for item in retrieved_chunks])
    prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate answer
    answer = rag_system.generate(prompt)
    
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer
    } 