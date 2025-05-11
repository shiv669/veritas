"""
RAG (Retrieval-Augmented Generation) implementation for Veritas

What this file does:
This is the heart of Veritas - it lets the AI answer questions by:
1. Finding relevant information in your documents (Retrieval)
2. Using that information to generate accurate answers (Generation)

Architecture Note:
- RAGSystem (this file) provides the core implementation used by MistralModel
- MistralModel (in run.py) serves as a wrapper that configures and uses RAGSystem
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
from .mps_utils import is_mps_available, clear_mps_cache, optimize_for_mps

logger = setup_logging(__name__)

class RAGSystem:
    """
    The main RAG system that combines retrieval and generation
    
    What it does:
    - Finds relevant chunks of text from your documents
    - Feeds those chunks to the AI to generate accurate answers
    - Keeps track of where information came from for citations
    
    Architecture Note:
    - This class implements the core RAG functionality
    - It is used by MistralModel in run.py, which serves as a configuration wrapper
    - This design separates core RAG logic (here) from application-specific logic (in MistralModel)
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
        
        Note:
        MistralModel configures these parameters based on user preferences and system constraints.
        """
        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load embedding model with standardized approach
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device,
            cache_folder=os.environ.get('TRANSFORMERS_CACHE')  # Use centralized SSD cache
        )
        
        # Load LLM with unified approach (same as run.py)
        self.llm_model_name = llm_model or Config.LLM_MODEL
        logger.info(f"Loading language model: {self.llm_model_name}")
        
        # Load tokenizer with efficient settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            trust_remote_code=True,
            use_fast=True,  # Use fast tokenizer when available
            padding_side="left",  # More efficient for generation
            cache_dir=os.environ.get('TRANSFORMERS_CACHE')  # Use SSD cache
        )
        
        # Load model directly with full precision (no legacy quantization attempts)
        logger.info("Loading model with full precision...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16,  # Use float16 for better performance
            device_map="auto",  # Automatic device mapping
            trust_remote_code=True,
            use_cache=True,  # Enable KV caching for faster generation
            cache_dir=os.environ.get('TRANSFORMERS_CACHE')  # Use SSD cache
        )
        
        # Setup generation pipeline with efficient config
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
            batch_size=1
        )
        
        # Apply MPS-specific optimizations if on Apple Silicon
        if self.device == "mps":
            self = optimize_for_mps(self)
            clear_mps_cache()
        
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
        chunks_npy = os.path.join(index_path, "chunks.npy")
        chunks_json = os.path.join(index_path, "chunks.json")
        
        logger.info(f"Looking for chunks at: {chunks_json}")
        
        if os.path.exists(chunks_npy):
            self.chunks = np.load(chunks_npy, allow_pickle=True).tolist()
        elif os.path.exists(chunks_json):
            import json
            with open(chunks_json, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        else:
            logger.warning(f"Chunks file not found (expected chunks.npy or chunks.json) in: {index_path}")
        
        # Load index - use the same directory as chunks
        index_file = os.path.join(index_path, "index.faiss")
        logger.info(f"Looking for index at: {index_file}")
        
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
    
    def get_retrieval_context(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get relevant context for a query, with detailed error handling
        
        Args:
            query: The query to get context for
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (combined context string, list of retrieved chunks)
        """
        try:
            # Ensure index is loaded with standard path
            if not self.index:
                # Use the absolute path to the index directory
                index_path = os.path.abspath(os.path.join(Config.MODELS_DIR, "faiss"))
                logger.info(f"Index not loaded, attempting to load from: {index_path}")
                
                index_file = os.path.join(index_path, "index.faiss")
                chunks_file = os.path.join(index_path, "chunks.json")
                
                if not os.path.exists(index_file):
                    return f"FAISS index file not found at: {index_file}", []
                
                if not os.path.exists(chunks_file):
                    return f"Chunks file not found at: {chunks_file}", []
                
                self.load_index(index_path)
            
            # Log retrieval attempt
            logger.info(f"Retrieving context for: {query[:50]}...")
            
            # Retrieve chunks
            results = self.retrieve(query, top_k=top_k)
            
            if not results:
                return "No relevant context found in the knowledge base for this query.", []
            
            # Log successful retrieval
            logger.info(f"Successfully retrieved {len(results)} chunks")
            
            # Combine retrieved chunks into context
            context = "\n\n".join([result["chunk"] for result in results])
            return context, results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error retrieving context: {str(e)}", []
    
    def process_large_context(self, context: str, max_chars: int = 3000) -> str:
        """
        Process large context to fit within token limits
        
        Args:
            context: The context to process
            max_chars: Maximum number of characters to keep
            
        Returns:
            Processed context
        """
        if len(context) <= max_chars:
            return context
            
        logger.info(f"Context is large ({len(context)} chars), chunking for better memory management")
        
        # Extract the most relevant paragraphs (up to max_chars chars)
        paragraphs = context.split("\n\n")
        
        # Take first paragraph (usually most relevant), then sample from the rest
        selected_text = paragraphs[0] if paragraphs else ""
        
        # If we have multiple paragraphs, select some from throughout the text
        if len(paragraphs) > 1:
            # Take evenly spaced samples from the document
            sample_count = min(5, len(paragraphs) - 1)
            sample_indices = [int(i * (len(paragraphs) - 1) / sample_count) for i in range(1, sample_count + 1)]
            
            for idx in sample_indices:
                if len(selected_text) < max_chars and idx < len(paragraphs):
                    selected_text += "\n\n" + paragraphs[idx]
        
        result = selected_text[:max_chars]  # Hard limit at max_chars
        logger.info(f"Reduced context to {len(result)} chars")
        return result
    
    def generate(self, prompt: str, 
                 max_new_tokens: int = 200,  # Reduced for better memory efficiency
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
        # Clear MPS cache before generation if using Apple Silicon
        if self.device == "mps":
            clear_mps_cache()
            
        with torch.inference_mode():
            result = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,  # Added to improve response quality
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Clear MPS cache after generation if using Apple Silicon
        if self.device == "mps":
            clear_mps_cache()
            
        return result[0]["generated_text"][len(prompt):]

    def generate_rag_response(self, 
                              query: str, 
                              top_k: int = 5, 
                              max_new_tokens: int = 200,
                              temperature: float = 0.7,
                              top_p: float = 0.9,
                              max_context_chars: int = 3000) -> Dict[str, Any]:
        """
        Generate a complete RAG response with both direct and context-augmented answers
        
        This is the core unified method that performs the entire RAG process in one call.
        It combines context retrieval, processing, and generation into a single workflow.
        
        Architectural role:
        - This is the primary method used by MistralModel.generate()
        - It's also used by the query_rag utility function
        - This centralization eliminates code duplication between run.py and rag.py
        
        Args:
            query: The user's query
            top_k: Number of chunks to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_context_chars: Maximum context length in characters
            
        Returns:
            Dictionary with context, direct response, and context-augmented response
        """
        # Get retrieval context
        context, retrieved_chunks = self.get_retrieval_context(query, top_k=top_k)
        
        # Generate direct answer (without context)
        direct_prompt = f"Question: {query}\n\nAnswer:"
        direct_response = self.generate(
            direct_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Process context if it's too large
        if isinstance(context, str) and len(context) > max_context_chars:
            context = self.process_large_context(context, max_chars=max_context_chars)
        
        # Generate combined response with context
        combined_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.

Context: {context}

Question: {query}

Answer:"""
        
        combined_response = self.generate(
            combined_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        return {
            "query": query,
            "context": context,
            "retrieved_chunks": retrieved_chunks,
            "direct_response": direct_response,
            "combined_response": combined_response
        }


def query_rag(query: str, rag_system: RAGSystem = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Process a query through the RAG system
    
    This function provides a simplified interface for external modules to access the RAG system
    without needing to interact with the full RAGSystem API. It serves as a convenience function
    for one-off queries that don't require maintaining a RAGSystem instance.
    
    Architectural role:
    - This is a standalone utility function (not a method of RAGSystem class)
    - It creates a temporary RAGSystem instance if one is not provided
    - It uses the more comprehensive RAGSystem.generate_rag_response internally
    - MistralModel in run.py uses RAGSystem directly for better efficiency and control
    
    Args:
        query: The user's query
        rag_system: RAG system instance or None to create a new one
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with query results including retrieved chunks and answer
    """
    # Create RAG system if not provided
    if rag_system is None:
        logger.info("Creating new RAG system")
        index_path = os.path.join(Config.MODELS_DIR, "faiss")
        logger.info(f"Using index path: {index_path}")
        rag_system = RAGSystem(index_path=index_path)
    
    # Use the centralized method to get relevant context and generate response
    result = rag_system.generate_rag_response(query, top_k=top_k)
    
    # For backward compatibility with existing code
    return {
        "query": query,
        "retrieved_chunks": result["retrieved_chunks"],
        "answer": result["combined_response"]
    } 