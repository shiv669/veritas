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
from typing import List, Dict, Any, Tuple, Optional, Union, cast
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from .config import Config, get_device
from .utils import setup_logging
from .mps_utils import is_mps_available, clear_mps_cache, optimize_for_mps
from .typing import ChunkType, ChunkList, QueryType, EmbeddingType, RetrievalResult, ModelOutput

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
                 embedding_model: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 index_path: Optional[str] = None,
                 device: Optional[str] = None):
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
        self.device: str = device or get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load embedding model with standardized approach
        self.embedding_model_name: str = embedding_model or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model: SentenceTransformer = SentenceTransformer(
            self.embedding_model_name,
            device=self.device,
            cache_folder=os.environ.get('TRANSFORMERS_CACHE')  # Use centralized SSD cache
        )
        
        # Load LLM with unified approach (same as run.py)
        self.llm_model_name: str = llm_model or Config.LLM_MODEL
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
        self.index: Optional[faiss.Index] = None
        self.chunks: ChunkList = []
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
    
    def embed_query(self, query: QueryType) -> EmbeddingType:
        """
        Embed query using the embedding model
        
        Args:
            query: Query to embed
            
        Returns:
            Query embedding
        """
        embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        return cast(EmbeddingType, embedding)
    
    def retrieve(self, query: QueryType, top_k: int = 5) -> List[RetrievalResult]:
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
        results: List[RetrievalResult] = []
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
    
    def get_retrieval_context(self, query: QueryType, top_k: int = 5) -> Tuple[str, List[RetrievalResult]]:
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
        
        # Simple truncation approach - more sophisticated methods could be used
        return context[:max_chars] + "..."
    
    def generate(self, prompt: str, 
                 max_new_tokens: int = 200,  # Reduced for better memory efficiency
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: The prompt to generate from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p for generation
            
        Returns:
            Generated text
        """
        try:
            # Log generation attempt
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            # Generate text
            result = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return str(generated_text)
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error generating text: {str(e)}"
    
    def generate_rag_response(self, 
                              query: QueryType, 
                              top_k: int = 5, 
                              max_new_tokens: int = 200,
                              temperature: float = 0.7,
                              top_p: float = 0.9,
                              max_context_chars: int = 3000) -> ModelOutput:
        """
        Generate a response using RAG
        
        Args:
            query: The query to answer
            top_k: Number of chunks to retrieve
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p for generation
            max_context_chars: Maximum number of characters for context
            
        Returns:
            Dictionary containing the response and related information
        """
        try:
            # Get context
            context, chunks = self.get_retrieval_context(query, top_k=top_k)
            
            # Process context to fit within token limits
            processed_context = self.process_large_context(context, max_chars=max_context_chars)
        
            # Generate direct response (without context)
            direct_prompt = f"USER: {query}\n\nASSISTANT:"
            direct_response = self.generate(
                direct_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        
            # Generate response with context
            rag_prompt = f"USER: Use the following information to answer the question:\n\nCONTEXT:\n{processed_context}\n\nQUESTION: {query}\n\nASSISTANT:"
            rag_response = self.generate(
                rag_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        
            # Return all information
            return {
                "query": query,
                "context": processed_context,
                "direct_response": direct_response,
                "combined_response": rag_response,
                "chunks": chunks
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "query": query,
                "context": f"Error: {str(e)}",
                "direct_response": f"Error generating response: {str(e)}",
                "combined_response": f"Error generating response: {str(e)}",
                "chunks": []
            }

def query_rag(query: QueryType, rag_system: Optional[RAGSystem] = None, top_k: int = 5) -> ModelOutput:
    """
    Query the RAG system
    
    This is a convenience function for querying the RAG system from other modules.
    It creates a new RAGSystem instance if one is not provided.
    
    Args:
        query: The query to answer
        rag_system: An existing RAGSystem instance (optional)
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary containing the response and related information
    """
    try:
        # Create RAGSystem if not provided
        if rag_system is None:
            rag_system = RAGSystem()
    
        # Generate response
        return rag_system.generate_rag_response(query, top_k=top_k)
    
    except Exception as e:
        logger.error(f"Error in query_rag: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a properly formatted error response
        return {
            "query": query,
            "context": f"Error: {str(e)}",
            "direct_response": f"Error generating response: {str(e)}",
            "combined_response": f"Error generating response: {str(e)}",
            "chunks": []
        } 