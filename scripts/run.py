#!/usr/bin/env python3
"""
run.py

Run Mistral model with support for terminal interface.
Optimized for M4 Mac to prevent kernel_task overload.

Architecture Overview:
- This script defines MistralModel, which serves as a wrapper around the core RAGSystem
- MistralModel configures RAGSystem with appropriate settings for the user's environment
- The relationship is: MistralModel (wrapper, this file) -> RAGSystem (core implementation, in rag.py)
- Now also includes the AI Scientist component for research tasks
"""

import json
import logging
from pathlib import Path
import os
from typing import Dict, List, Optional, Union, Any, Tuple, cast, Type, Set, Callable
from dataclasses import dataclass
from enum import Enum
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline
import re
import psutil  # For better process management
import gc
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import subprocess
import time

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.veritas.config import Config, get_device
from src.veritas.mps_utils import is_mps_available, clear_mps_cache, optimize_memory_for_m4
from src.veritas.typing import ModelOutput, QueryType

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

class SystemMode(Enum):
    """Supported system modes."""
    RAG = "rag"
    AI_SCIENTIST = "ai_scientist"

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
        self.config: ModelConfig = config
        self.rag_system: Optional[Any] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generator: Optional[Pipeline] = None
        
        # Set number of threads for CPU operations
        if self.config.device == "cpu":
            torch.set_num_threads(self.config.num_threads)
    
    def load(self) -> None:
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
            if self.rag_system is None:
                return "RAG system not initialized"
                
            context, _ = self.rag_system.get_retrieval_context(
                prompt, 
                top_k=self.config.max_retrieved_chunks
            )
            
            if isinstance(context, str):
                return context
            else:
                return f"Error: unexpected context type {type(context)}"
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error retrieving context: {str(e)}"
    
    def generate(self, prompt: str) -> Tuple[str, str, str]:
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
            result: ModelOutput = self.rag_system.generate_rag_response(
                query=prompt,
                top_k=self.config.max_retrieved_chunks,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_context_chars=3000  # Hard limit for M4 optimization
            )
            
            context = result.get("context", "")
            direct_response = result.get("direct_response", "")
            combined_response = result.get("combined_response", "")
            
            return context, direct_response, combined_response
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            # Force memory cleanup on error
            if self.config.device == "mps":
                clear_mps_cache()
            
            # Return error messages instead of raising
            error_msg = f"Error generating response: {str(e)}"
            return error_msg, error_msg, error_msg

class AIScientistInterface:
    """Interface for the AI Scientist component."""
    
    def __init__(self):
        self.ai_scientist_path: str = os.path.join(PROJECT_ROOT, "src/veritas/ai_scientist")
        self.setup_paths()
    
    def setup_paths(self) -> None:
        """Set up Python paths for AI Scientist."""
        veritas_path: str = PROJECT_ROOT
        templates_path: str = os.path.join(PROJECT_ROOT, "models", "Cognition")
        
        # Add to Python path if not already there
        if veritas_path not in sys.path:
            sys.path.insert(0, veritas_path)
        if templates_path not in sys.path:
            sys.path.insert(0, templates_path)
    
    def list_available_templates(self) -> List[str]:
        """
        List available research templates.
        
        Returns:
            List of template names
        """
        templates_dir: str = os.path.join(PROJECT_ROOT, "models", "Cognition", "templates")
        templates: List[str] = []
        
        try:
            templates = [d for d in os.listdir(templates_dir) 
                        if os.path.isdir(os.path.join(templates_dir, d))]
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            templates = ["nanoGPT_lite"]  # fallback
        
        return templates
    
    def run_interactive(self) -> bool:
        """
        Run the AI Scientist in interactive mode.
        
        Returns:
            True if successful, False otherwise
        """
        script_path: str = os.path.join(self.ai_scientist_path, "run_interface.py")
        result = subprocess.run([sys.executable, script_path], check=False)
        return result.returncode == 0
    
    def run_simple_test(self) -> bool:
        """
        Run the simple test script.
        
        Returns:
            True if successful, False otherwise
        """
        print("Running simple test mode...")
        script_path: str = os.path.join(self.ai_scientist_path, "test_simple.py")
        result = subprocess.run([sys.executable, script_path], check=False)
        return result.returncode == 0
    
    def run_with_params(self, mode: str, experiment: str, num_ideas: int) -> bool:
        """
        Run the AI Scientist with specific parameters.
        
        Args:
            mode: Mode to run in ("direct" or "full")
            experiment: Name of the experiment template
            num_ideas: Number of ideas to generate
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Running {mode} mode for {experiment}, generating {num_ideas} idea(s)...")
        script_path: str = os.path.join(self.ai_scientist_path, "run_scientist.py")
        
        cmd: List[str] = [
            sys.executable, 
            script_path, 
            "--phase", "idea", 
            "--experiment", experiment, 
            "--num-ideas", str(num_ideas)
        ]
        
        if mode == "direct":
            cmd.append("--use-direct-implementation")
            
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0

class TerminalUI:
    """High-performance terminal-based user interface."""
    
    def __init__(self, model: MistralModel):
        self.model: MistralModel = model
        self.ai_scientist: AIScientistInterface = AIScientistInterface()
    
    def run(self, mode: SystemMode = SystemMode.RAG, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        """
        Run the terminal interface.
        
        Args:
            mode: System mode to run in (RAG or AI Scientist)
            host: Host to bind to (for API mode)
            port: Port to bind to (for API mode)
        """
        if mode == SystemMode.RAG:
            self.run_rag_interface()
        elif mode == SystemMode.AI_SCIENTIST:
            self.run_ai_scientist_interface()
        else:
            print(f"Unknown mode: {mode}")
            # Handle unknown mode gracefully
            self.run_rag_interface()  # Default to RAG interface
    
    def run_rag_interface(self) -> None:
        """Run the RAG terminal interface."""
        print("\n" + "="*70)
        print("  MISTRAL RAG SYSTEM - OPTIMIZED FOR M4 MAX (128GB RAM)")
        print("="*70 + "\n")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'gc' to force garbage collection if needed.")
        print("Type 'scientist' to switch to AI Scientist mode.")
        print("\n")
        
        while True:
            try:
                prompt: str = input("\n📝 Enter your prompt: ").strip()
                
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                if prompt.lower() == 'scientist':
                    self.run_ai_scientist_interface()
                    # After returning from AI Scientist mode, show the RAG welcome message again
                    print("\n" + "="*70)
                    print("  MISTRAL RAG SYSTEM - OPTIMIZED FOR M4 MAX (128GB RAM)")
                    print("="*70 + "\n")
                    print("Type 'exit' or 'quit' to end the session.")
                    print("Type 'gc' to force garbage collection if needed.")
                    print("Type 'scientist' to switch to AI Scientist mode.")
                    print("\n")
                    continue
                    
                if prompt.lower() == 'gc':
                    print("Forcing garbage collection...")
                    gc.collect()
                    if self.model.config.device == "mps":
                        clear_mps_cache()
                    print("Memory cleaned.")
                    continue
                
                if not prompt:
                    continue
                
                print("\nGenerating response...")
                start_time: float = time.time()
                
                # Get the context from the RAG system
                context: str = self.model.get_retrieval_context(prompt)
                
                # Generate the direct and combined responses
                context, direct_response, combined_response = self.model.generate(prompt)
                
                elapsed_time: float = time.time() - start_time
                
                # Display responses in a formatted way
                print("\n" + "="*70)
                print("  RETRIEVED CONTEXT")
                print("="*70)
                print(context or "No relevant context found.")
                
                print("\n" + "="*70)
                print("  DIRECT RESPONSE")
                print("="*70)
                print(direct_response)
                
                print("\n" + "="*70)
                print("  COMBINED RESPONSE")
                print("="*70)
                print(combined_response)
                
                print(f"\nGenerated in {elapsed_time:.2f} seconds.")
                
                # Force memory cleanup after generation
                if self.model.config.device == "mps":
                    clear_mps_cache()
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                break
                
            except Exception as e:
                import traceback
                print(f"Error: {str(e)}")
                logger.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Force memory cleanup on error
                if self.model.config.device == "mps":
                    clear_mps_cache()
    
    def run_ai_scientist_interface(self) -> None:
        """Run the AI Scientist interface."""
        print("\n" + "="*70)
        print("  VERITAS AI SCIENTIST - RESEARCH ASSISTANT")
        print("="*70 + "\n")
        print("Select a mode to run:")
        print("  1. Interactive Mode (Guided UI)")
        print("  2. Simple Test (Quick Demo)")
        print("  3. Run with Parameters (Advanced)")
        print("  4. Back to RAG System")
        print("\n")
        
        while True:
            try:
                choice: str = input("\nSelect mode (1-4): ").strip()
                
                if choice == '1':
                    print("\nLaunching interactive mode...")
                    self.ai_scientist.run_interactive()
                    break
                
                elif choice == '2':
                    print("\nRunning simple test...")
                    self.ai_scientist.run_simple_test()
                    break
                
                elif choice == '3':
                    # List available templates
                    templates: List[str] = self.ai_scientist.list_available_templates()
                    print("\nAvailable research templates:")
                    for i, template in enumerate(templates, 1):
                        print(f"  {i}. {template}")
                    
                    # Select template
                    while True:
                        try:
                            choice = input(f"\nSelect template (1-{len(templates)}) [default: 1]: ")
                            if not choice:
                                template_idx = 0
                                break
                            
                            template_idx = int(choice) - 1
                            if 0 <= template_idx < len(templates):
                                break
                            else:
                                print(f"Please enter a number between 1 and {len(templates)}.")
                        except ValueError:
                            print("Please enter a valid number.")
                    
                    experiment: str = templates[template_idx]
                    
                    # Select number of ideas
                    while True:
                        try:
                            choice = input("\nNumber of ideas to generate (1-5) [default: 1]: ")
                            if not choice:
                                num_ideas = 1
                                break
                            
                            num_ideas = int(choice)
                            if 1 <= num_ideas <= 5:
                                break
                            else:
                                print("Please enter a number between 1 and 5.")
                        except ValueError:
                            print("Please enter a valid number.")
                    
                    # Select implementation
                    print("\nSelect implementation:")
                    print("  1. Optimized (recommended)")
                    print("  2. Comprehensive (in-depth)")
                    
                    impl_choice: str = input("\nSelect implementation (1-2) [default: 1]: ")
                    mode: str = "direct" if not impl_choice or impl_choice == "1" else "full"
                    
                    self.ai_scientist.run_with_params(mode, experiment, num_ideas)
                    break
                    
                elif choice == '4':
                    print("\nReturning to RAG System...")
                    return
                
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Returning to RAG System...")
                return
                
            except Exception as e:
                import traceback
                print(f"Error: {str(e)}")
                logger.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())

def run_model(
    ui_framework: UIFramework,
    deployment_mode: DeploymentMode,
    system_mode: SystemMode = SystemMode.RAG,
    model_config: Optional[ModelConfig] = None,
    host: str = "0.0.0.0",
    port: Optional[int] = None
) -> None:
    """
    Run the Mistral model with the specified UI framework and deployment mode.
    
    Args:
        ui_framework: The UI framework to use
        deployment_mode: The deployment mode
        system_mode: The system mode (RAG or AI Scientist)
        model_config: Configuration for the model
        host: The host to bind to
        port: The port to bind to
    """
    if model_config is None:
        model_config = ModelConfig()
    
    if system_mode == SystemMode.AI_SCIENTIST:
        # For AI Scientist mode, we can skip loading the model
        # and directly launch the AI Scientist interface
        if ui_framework == UIFramework.TERMINAL:
            # Create a temporary UI with a placeholder model
            # This is a bit of a hack, but it allows us to reuse the existing UI code
            terminal_ui = TerminalUI(MistralModel(model_config))
            terminal_ui.run_ai_scientist_interface()
            return
        else:
            raise ValueError(f"Unsupported UI framework for AI Scientist: {ui_framework}")
    
    # We only reach here for RAG mode
    try:
        # Create and load the model
        model: MistralModel = MistralModel(model_config)
        logger.info("Loading model...")
        model.load()
        logger.info("Model loaded successfully.")
        
        # Run with the specified UI framework and deployment mode
        if ui_framework == UIFramework.TERMINAL:
            ui: TerminalUI = TerminalUI(model)
            ui.run(system_mode, host, port)
        else:
            raise ValueError(f"Unsupported UI framework: {ui_framework}")
    
    except Exception as e:
        import traceback
        logger.error(f"Failed to run model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Mistral model.")
    parser.add_argument(
        "--ui", 
        type=str, 
        default="terminal", 
        choices=["terminal"],
        help="UI framework to use"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="local", 
        choices=["local", "api", "docker"],
        help="Deployment mode"
    )
    parser.add_argument(
        "--system", 
        type=str, 
        default="rag", 
        choices=["rag", "ai_scientist"],
        help="System mode to use (RAG or AI Scientist)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Port to bind to (for API mode)"
    )
    
    args = parser.parse_args()
    
    # Convert string arguments to enum values
    ui_framework: UIFramework = UIFramework(args.ui)
    deployment_mode: DeploymentMode = DeploymentMode(args.mode)
    system_mode: SystemMode = SystemMode(args.system)
    
    try:
        # Run the model
        run_model(
            ui_framework=ui_framework,
            deployment_mode=deployment_mode,
            system_mode=system_mode,
            host=args.host,
            port=args.port
        )
    except Exception as e:
        logger.error(f"Failed to run model: {str(e)}")
        sys.exit(1)
