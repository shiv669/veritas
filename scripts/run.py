#!/usr/bin/env python3
"""
run.py

Run Mistral model with support for terminal interface.
Optimized for M4 Mac to prevent kernel_task overload.
"""

import json
import logging
from pathlib import Path
import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import psutil  # For better process management
import gc
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.veritas.config import Config, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOGS_DIR, "mistral.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create temporary directory on SSD with absolute path
SSD_TEMP_DIR = "/Volumes/8SSD/veritas/tmp"
os.makedirs(SSD_TEMP_DIR, exist_ok=True)
logger.info(f"Using temporary directory: {SSD_TEMP_DIR}")

# Set environment variables to limit resource usage - Optimized for M4 Mac with 128GB RAM
os.environ.update({
    'OMP_NUM_THREADS': '4',  # Allow more OpenMP threads for M4
    'MKL_NUM_THREADS': '4',  # Allow more MKL threads for M4
    'NUMEXPR_NUM_THREADS': '4',  # Allow more NumExpr threads for M4
    'TOKENIZERS_PARALLELISM': 'true',  # Enable tokenizer parallelism for speed
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable upper limit to prevent OOM
    'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.6',  # Keep more in memory
    'PYTORCH_MPS_MEMORY_LIMIT': '100GB',  # Use more of available RAM
    'PYTORCH_ENABLE_MPS_FALLBACK': '1',  # Enable fallback
    'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',  # Reduce warnings
    'TMPDIR': SSD_TEMP_DIR,  # Set temp directory to SSD instead of Macintosh HD
    'TORCH_HOME': os.path.join(SSD_TEMP_DIR, 'torch'),  # Store PyTorch cache on SSD
    'TRANSFORMERS_CACHE': os.path.join(SSD_TEMP_DIR, 'transformers'),  # Store Transformers cache on SSD
    'HF_HOME': os.path.join(SSD_TEMP_DIR, 'huggingface'),  # Store HuggingFace cache on SSD
})

# Set process priority at the start
p = psutil.Process(os.getpid())
p.nice(10)  # Lower but not too low (0-19, higher is lower priority)

class UIFramework(Enum):
    """Supported UI frameworks."""
    TERMINAL = "terminal"

class DeploymentMode(Enum):
    """Supported deployment modes."""
    LOCAL = "local"
    API = "api"
    DOCKER = "docker"

@dataclass
class ModelConfig:
    """Configuration for Mistral model."""
    model_name: str = Config.LLM_MODEL
    max_new_tokens: int = 200  # Further reduced for better memory efficiency
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    device: str = get_device()
    # M4-specific optimizations
    torch_dtype: torch.dtype = torch.float16  # Use float16 for better performance
    use_cache: bool = True
    num_threads: int = 8  # Use more threads for M4
    batch_size: int = 1
    max_context_length: int = 1024  # Reduced context length for memory efficiency
    max_retrieved_chunks: int = 2  # Further reduced for better memory efficiency

class MistralModel:
    """Wrapper for Mistral model."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model with configuration."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Set number of threads for CPU operations
        if self.config.device == "cpu":
            torch.set_num_threads(self.config.num_threads)
    
    def load(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model {self.config.model_name}...")
            
            # Load tokenizer with efficient settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                use_fast=True,  # Use fast tokenizer when available
                padding_side="left",  # More efficient for generation
                cache_dir=os.environ.get('TRANSFORMERS_CACHE')  # Use SSD cache
            )
            
            # Skip quantization attempts and load directly with full precision
            logger.info("Loading with full precision...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                use_cache=self.config.use_cache,
                cache_dir=os.environ.get('TRANSFORMERS_CACHE')  # Use SSD cache
            )
            
            # Setup generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=self.config.torch_dtype,
                batch_size=self.config.batch_size
            )
            
            logger.info("Model loaded successfully")
            # Force memory cleanup after loading
            if self.config.device == "mps":
                torch.mps.empty_cache()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_retrieval_context(self, prompt: str) -> str:
        """Get relevant context for the prompt from the RAG system."""
        try:
            # Ensure directories exist
            Config.ensure_dirs()
            
            # Use the absolute path to the index directory
            index_path = os.path.abspath(os.path.join(Config.MODELS_DIR, "faiss"))
            logger.info(f"Using index path: {index_path}")
            
            # Check if index exists with detailed logging
            index_file = os.path.join(index_path, "index.faiss")
            chunks_file = os.path.join(index_path, "chunks.json")
            
            logger.info(f"Checking for index file: {index_file} (exists: {os.path.exists(index_file)})")
            logger.info(f"Checking for chunks file: {chunks_file} (exists: {os.path.exists(chunks_file)})")
            
            if not os.path.exists(index_file):
                return "FAISS index file not found at: " + index_file
            
            if not os.path.exists(chunks_file):
                return "Chunks file not found at: " + chunks_file
            
            # Initialize RAG system with direct file paths
            from src.veritas.rag import RAGSystem
            rag = RAGSystem(
                embedding_model=Config.EMBEDDING_MODEL,
                llm_model=self.config.model_name,
                index_path=index_path,
                device=self.config.device
            )
            
            # Log retrieval attempt
            logger.info(f"Retrieving context for: {prompt[:50]}...")
            
            # Retrieve chunks with your existing system
            results = rag.retrieve(prompt, top_k=self.config.max_retrieved_chunks)
            
            if not results:
                return "No relevant context found in the knowledge base for this query."
            
            # Log successful retrieval
            logger.info(f"Successfully retrieved {len(results)} chunks")
            
            # Combine retrieved chunks into context
            context = "\n\n".join([result["chunk"] for result in results])
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error retrieving context: {str(e)}"
    
    def generate(self, prompt: str) -> tuple[str, str, str]:
        """Generate text from prompt with high-performance settings."""
        if not self.model or not self.tokenizer or not self.generator:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Get context from RAG with optimized settings
            context = self.get_retrieval_context(prompt)
            
            # Memory cleanup before generation
            if self.config.device == "mps":
                torch.mps.empty_cache()
            
            # Generate direct answer without context
            direct_prompt = f"Question: {prompt}\n\nAnswer:"
            with torch.inference_mode():
                direct_result = self.generator(
                    direct_prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=self.config.do_sample,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                direct_response = direct_result[0]["generated_text"][len(direct_prompt):]
            
            # Memory cleanup between generations
            if self.config.device == "mps":
                torch.mps.empty_cache()
            
            # Process context to reduce its size if needed
            if len(context) > 4000:
                logger.info(f"Context is large ({len(context)} chars), chunking for better memory management")
                
                # More aggressive chunking to ensure we don't overwhelm the model
                # Extract the most relevant paragraphs (up to 3000 chars)
                paragraphs = context.split("\n\n")
                
                # Take first paragraph (usually most relevant), then sample from the rest
                selected_text = paragraphs[0] if paragraphs else ""
                
                # If we have multiple paragraphs, select some from throughout the text
                if len(paragraphs) > 1:
                    # Take evenly spaced samples from the document
                    sample_count = min(5, len(paragraphs) - 1)
                    sample_indices = [int(i * (len(paragraphs) - 1) / sample_count) for i in range(1, sample_count + 1)]
                    
                    for idx in sample_indices:
                        if len(selected_text) < 3000 and idx < len(paragraphs):
                            selected_text += "\n\n" + paragraphs[idx]
                
                context = selected_text[:3000]  # Hard limit at 3000 chars
                logger.info(f"Reduced context to {len(context)} chars")
            
            # Generate combined response with context
            combined_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.

Context: {context}

Question: {prompt}

Answer:"""
            
            with torch.inference_mode():
                combined_result = self.generator(
                    combined_prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=self.config.do_sample,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                combined_response = combined_result[0]["generated_text"][len(combined_prompt):]
            
            return context, direct_response, combined_response
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            # Force memory cleanup on error
            if self.config.device == "mps":
                torch.mps.empty_cache()
            elif self.config.device == "cuda":
                torch.cuda.empty_cache()
            raise

class TerminalUI:
    """High-performance terminal-based user interface."""
    
    def __init__(self, model: MistralModel):
        self.model = model
    
    def run(self, host: str = "0.0.0.0", port: int = None):
        """Run the terminal interface."""
        print("\n" + "="*70)
        print("  MISTRAL RAG SYSTEM - OPTIMIZED FOR M4 MAX (128GB RAM)")
        print("="*70 + "\n")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'gc' to force garbage collection if needed.")
        print("\n")
        
        while True:
            try:
                prompt = input("\n📝 Enter your prompt: ").strip()
                
                if prompt.lower() in ['exit', 'quit']:
                    break
                    
                if prompt.lower() == 'gc':
                    print("Forcing garbage collection...")
                    gc.collect()
                    if self.model.config.device == "mps":
                        torch.mps.empty_cache()
                    print("Memory cleaned.")
                    continue
                
                if not prompt:
                    continue
                
                # Process the prompt
                print("\n⏳ Processing query (using high-performance settings)...\n")
                context, direct_response, combined_response = self.model.generate(prompt)
                
                # Display all three parts with nice formatting
                print("\n" + "="*70)
                print("  1. RETRIEVED CONTEXT")
                print("="*70)
                print(context)
                
                print("\n" + "="*70)
                print("  2. MISTRAL'S DIRECT ANSWER")
                print("="*70)
                print(direct_response)
                
                print("\n" + "="*70)
                print("  3. FINAL RESPONSE (CONTEXT + QUERY)")
                print("="*70)
                print(combined_response)
                print("="*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                # Provide option to continue after error
                if input("\nContinue? (y/n): ").lower() != 'y':
                    break

def run_model(
    ui_framework: UIFramework,
    deployment_mode: DeploymentMode,
    model_config: Optional[ModelConfig] = None,
    host: str = "0.0.0.0",
    port: Optional[int] = None
) -> None:
    """
    Run the Mistral model with the terminal UI.
    
    Args:
        ui_framework: UI framework to use (only TERMINAL is supported)
        deployment_mode: Deployment mode
        model_config: Model configuration
        host: Host to bind to (not used for terminal UI)
        port: Port to bind to (not used for terminal UI)
    """
    try:
        # Initialize model
        model_config = model_config or ModelConfig()
        model = MistralModel(model_config)
        
        # Clean memory before loading
        gc.collect()
        if model_config.device == "mps":
            torch.mps.empty_cache()
            
        model.load()
        
        # Only terminal UI is supported
        if ui_framework != UIFramework.TERMINAL:
            logger.warning(f"Unsupported UI framework: {ui_framework}. Using Terminal UI instead.")
            
        ui = TerminalUI(model)
        ui.run()
    except Exception as e:
        logger.error(f"Error in run_model: {str(e)}")
        # Final cleanup
        gc.collect()
        if model_config and model_config.device == "mps":
            torch.mps.empty_cache()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Mistral model")
    parser.add_argument("--ui-framework", type=str,
                      choices=[f.value for f in UIFramework],
                      default=UIFramework.TERMINAL.value,
                      help="UI framework to use")
    parser.add_argument("--deployment-mode", type=str,
                      choices=[m.value for m in DeploymentMode],
                      default=DeploymentMode.LOCAL.value,
                      help="Deployment mode")
    parser.add_argument("--model-name", type=str,
                      default=Config.LLM_MODEL,
                      help="Name of the model to use")
    parser.add_argument("--host", type=str,
                      default="0.0.0.0",
                      help="Host to bind to (not used for terminal UI)")
    parser.add_argument("--port", type=int,
                      help="Port to bind to (not used for terminal UI)")
    
    args = parser.parse_args()
    
    # Create model config
    model_config = ModelConfig(model_name=args.model_name)
    
    # Run model with terminal UI
    run_model(
        UIFramework(args.ui_framework),
        DeploymentMode(args.deployment_mode),
        model_config,
        args.host,
        args.port
    )
