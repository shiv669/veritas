#!/usr/bin/env python3
"""
run.py

Run Mistral model with support for terminal interface.
Optimized for M4 Mac to prevent kernel_task overload.

Architecture Overview:
- This script defines MistralModel, which serves as a wrapper around the core RAGSystem
- MistralModel configures RAGSystem with appropriate settings for the user's environment
- The relationship is: MistralModel (wrapper, this file) -> RAGSystem (core implementation, in rag.py)
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
from src.veritas.mps_utils import is_mps_available, clear_mps_cache, optimize_memory_for_m4

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

# Create directories if they don't exist and setup environment
Config.ensure_dirs()

# Apply comprehensive M4 optimizations (both environment and runtime)
optimize_memory_for_m4()

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
    num_threads: int = Config.CPU_THREADS  # Use centralized configuration
    batch_size: int = 1
    max_context_length: int = 1024  # Reduced context length for memory efficiency
    max_retrieved_chunks: int = 2  # Further reduced for better memory efficiency

class MistralModel:
    """
    Wrapper for Mistral model that uses RAGSystem internally.
    
    Architecture Note:
    - MistralModel is a high-level wrapper around the core RAGSystem implementation
    - It configures RAGSystem with appropriate settings for the user's environment
    - It handles application-specific concerns like configuration and error handling
    - The core RAG functionality is delegated to the RAGSystem class in src/veritas/rag.py
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model with configuration.
        
        Args:
            config: Configuration for the model including parameters 
                   that will be passed to the underlying RAGSystem
        """
        self.config = config
        self.rag_system = None
        
        # Set number of threads for CPU operations
        if self.config.device == "cpu":
            torch.set_num_threads(self.config.num_threads)
    
    def load(self):
        """
        Load the model and tokenizer using RAGSystem.
        
        This method creates a RAGSystem instance configured with the
        parameters from this class's ModelConfig, then exposes the
        RAGSystem's model, tokenizer, and generator for compatibility.
        """
        try:
            logger.info(f"Loading model {self.config.model_name}...")
            
            # Import RAGSystem (inside method to avoid circular imports)
            from src.veritas.rag import RAGSystem
            
            # Create RAGSystem instance with our configuration
            self.rag_system = RAGSystem(
                embedding_model=Config.EMBEDDING_MODEL,
                llm_model=self.config.model_name,
                device=self.config.device
            )
            
            # For compatibility with the existing interface
            self.model = self.rag_system.model
            self.tokenizer = self.rag_system.tokenizer
            self.generator = self.rag_system.generator
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_retrieval_context(self, prompt: str) -> str:
        """
        Get relevant context for the prompt from the RAG system.
        
        This is a wrapper method that delegates to RAGSystem.get_retrieval_context
        while handling Veritas-specific concerns like directory setup and error logging.
        
        Args:
            prompt: The user's query or prompt
            
        Returns:
            Relevant context extracted from the knowledge base
        """
        try:
            # Ensure directories exist
            Config.ensure_dirs()
            
            # Log the operation
            logger.info(f"Getting retrieval context for: {prompt[:50]}...")
            
            # Use the centralized method from RAGSystem
            context, _ = self.rag_system.get_retrieval_context(
                prompt, 
                top_k=self.config.max_retrieved_chunks
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error retrieving context: {str(e)}"
    
    def generate(self, prompt: str) -> tuple[str, str, str]:
        """
        Generate text from prompt with high-performance settings.
        
        This method delegates to the RAGSystem.generate_rag_response method
        while handling Veritas-specific concerns like error handling and memory cleanup.
        
        Args:
            prompt: The user's query or prompt
            
        Returns:
            Tuple of (retrieved context, direct response, context-augmented response)
        """
        if not self.rag_system or not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Use the centralized method for complete RAG response
            result = self.rag_system.generate_rag_response(
                query=prompt,
                top_k=self.config.max_retrieved_chunks,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_context_chars=3000  # Hard limit for M4 optimization
            )
            
            return result["context"], result["direct_response"], result["combined_response"]
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            # Force memory cleanup on error
            if self.config.device == "mps":
                clear_mps_cache()
            
            # Let the exception propagate
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
                        clear_mps_cache()
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
            clear_mps_cache()
            
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
            clear_mps_cache()

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
