#!/usr/bin/env python3
"""
run.py

Run Mistral model with support for different UI frameworks and deployment options.
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
import multiprocessing as mp
from multiprocessing import Process, Queue, cpu_count
import psutil  # For better process management
import torch.multiprocessing as torch_mp  # Use PyTorch's multiprocessing

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

# Set environment variables to limit resource usage
os.environ.update({
    'OMP_NUM_THREADS': '1',  # Limit OpenMP threads
    'MKL_NUM_THREADS': '1',  # Limit MKL threads
    'NUMEXPR_NUM_THREADS': '1',  # Limit NumExpr threads
    'TOKENIZERS_PARALLELISM': 'false',  # Disable tokenizer parallelism
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable automatic growth
    'PYTORCH_MPS_MEMORY_LIMIT': '102GB',  # 80% of 128GB
    'PYTORCH_ENABLE_MPS_FALLBACK': '1',  # Enable fallback for stability
})

# Set process priority at the start
p = psutil.Process(os.getpid())
p.nice(15)  # Even lower priority for main process

class UIFramework(Enum):
    """Supported UI frameworks."""
    TERMINAL = "terminal"
    STREAMLIT = "streamlit"
    FLASK = "flask"
    FASTAPI = "fastapi"

class DeploymentMode(Enum):
    """Supported deployment modes."""
    LOCAL = "local"
    API = "api"
    DOCKER = "docker"

@dataclass
class ModelConfig:
    """Configuration for Mistral model."""
    model_name: str = Config.LLM_MODEL
    max_new_tokens: int = 256  # Reduced from 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    device: str = get_device()
    # M4-specific optimizations
    torch_dtype: torch.dtype = torch.float32 if get_device() == "mps" else torch.float16
    use_cache: bool = True
    num_threads: int = 1
    batch_size: int = 1
    max_context_length: int = 1024  # Limit context length

class MistralModel:
    """Wrapper for Mistral model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Set number of threads for CPU operations
        if self.config.device == "cpu":
            torch.set_num_threads(self.config.num_threads)
    
    def load(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            use_cache=self.config.use_cache,
            low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
        ).to(self.config.device)
        
        # Enable model optimizations
        if self.config.device == "mps":
            self.model = self.model.to("mps")
            # Enable MPS optimizations
            torch.mps.empty_cache()  # Clear MPS cache
        elif self.config.device == "cuda":
            self.model = self.model.cuda()
            torch.cuda.empty_cache()  # Clear CUDA cache
        
        logger.info("Model loaded successfully")
    
    def get_retrieval_context(self, prompt: str) -> str:
        """Get relevant context for the prompt from the RAG system."""
        # For now, return a simple placeholder context
        # In a real implementation, this would query your RAG system
        return "This is a placeholder context. In a real implementation, this would contain relevant information retrieved from your knowledge base."
    
    def generate(self, prompt: str) -> tuple[str, str, str]:
        """Generate text from prompt and return context, direct answer, and combined response."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Get retrieval context (placeholder for now)
            context = "This is a placeholder context. In a real implementation, this would contain relevant information retrieved from your knowledge base."
            
            # Generate direct answer without context
            direct_prompt = f"Question: {prompt}\n\nAnswer:"
            direct_inputs = self.tokenizer(direct_prompt, return_tensors="pt", truncation=True, max_length=self.config.max_context_length).to(self.config.device)
            
            with torch.inference_mode():
                direct_outputs = self.model.generate(
                    **direct_inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            direct_response = self.tokenizer.decode(direct_outputs[0], skip_special_tokens=True)
            
            # Generate combined response with context
            combined_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
            combined_inputs = self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, max_length=self.config.max_context_length).to(self.config.device)
            
            with torch.inference_mode():
                combined_outputs = self.model.generate(
                    **combined_inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            combined_response = self.tokenizer.decode(combined_outputs[0], skip_special_tokens=True)
            
            # Clean up memory
            del direct_outputs, combined_outputs
            del direct_inputs, combined_inputs
            if self.config.device == "mps":
                torch.mps.empty_cache()
            elif self.config.device == "cuda":
                torch.cuda.empty_cache()
            
            return context, direct_response, combined_response
        
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            if self.config.device == "mps":
                torch.mps.empty_cache()
            elif self.config.device == "cuda":
                torch.cuda.empty_cache()
            raise

class StreamlitUI:
    """Streamlit-based user interface."""
    
    def __init__(self, model: MistralModel):
        self.model = model
    
    def run(self, host: str = "0.0.0.0", port: int = 8501):
        """Run the Streamlit interface."""
        try:
            import streamlit as st
        except ImportError:
            logger.error("Streamlit is required for this UI framework")
            raise
        
        st.title("Mistral Chat")
        st.write("Chat with the Mistral model")
        
        prompt = st.text_area("Enter your prompt:", height=100)
        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating..."):
                    response = self.model.generate(prompt)
                st.write("Response:")
                st.write(response)
            else:
                st.warning("Please enter a prompt")

class FlaskAPI:
    """Flask-based API server."""
    
    def __init__(self, model: MistralModel):
        self.model = model
    
    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """Run the Flask API server."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("Flask is required for this API framework")
            raise
        
        app = Flask(__name__)
        
        @app.route("/generate", methods=["POST"])
        def generate():
            data = request.get_json()
            prompt = data.get("prompt")
            if not prompt:
                return jsonify({"error": "No prompt provided"}), 400
            
            response = self.model.generate(prompt)
            return jsonify({"response": response})
        
        app.run(host=host, port=port)

class FastAPI:
    """FastAPI-based API server."""
    
    def __init__(self, model: MistralModel):
        self.model = model
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            import uvicorn
        except ImportError:
            logger.error("FastAPI and uvicorn are required for this API framework")
            raise
        
        app = FastAPI(title="Mistral API")
        
        class GenerateRequest(BaseModel):
            prompt: str
        
        @app.post("/generate")
        async def generate(request: GenerateRequest):
            response = self.model.generate(request.prompt)
            return {"response": response}
        
        uvicorn.run(app, host=host, port=port)

class ModelWorker(Process):
    """Worker process for model inference."""
    def __init__(self, model_config, input_queue, output_queue):
        super().__init__()
        self.model_config = model_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model = None
        self.tokenizer = None
        self.daemon = True
    
    def run(self):
        """Run the worker process."""
        try:
            # Set process priority
            p = psutil.Process(os.getpid())
            p.nice(20)  # Even lower priority for workers
            
            # Initialize model in the worker process
            self.model = MistralModel(self.model_config)
            self.model.load()
            
            while True:
                task = self.input_queue.get()
                if task is None:
                    break
                
                prompt = task
                try:
                    # Generate response with strict memory management
                    with torch.inference_mode():
                        response = self.model.generate(prompt)
                    self.output_queue.put(("success", response))
                except Exception as e:
                    self.output_queue.put(("error", str(e)))
                
                # Aggressive memory cleanup
                import gc
                gc.collect()
                if self.model_config.device == "mps":
                    torch.mps.empty_cache()
                elif self.model_config.device == "cuda":
                    torch.cuda.empty_cache()
        except Exception as e:
            self.output_queue.put(("error", f"Worker error: {str(e)}"))
        finally:
            del self.model
            del self.tokenizer
            if self.model_config.device == "mps":
                torch.mps.empty_cache()
            elif self.model_config.device == "cuda":
                torch.cuda.empty_cache()

class TerminalUI:
    """Simple terminal-based user interface."""
    
    def __init__(self, model: MistralModel):
        self.model = model
        self.num_workers = 1  # Use only one worker
    
    def run(self, host: str = "0.0.0.0", port: int = None):
        """Run the terminal interface."""
        print("Welcome to Mistral Chat! (Using 1 worker)")
        print("Type 'exit' or 'quit' to end the session.")
        print()
        
        while True:
            try:
                prompt = input("Enter your prompt: ").strip()
                
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                if not prompt:
                    continue
                
                # Get context, direct answer, and combined response
                context, direct_response, combined_response = self.model.generate(prompt)
                
                # Display all three parts
                print("\n1. Retrieved Context:")
                print("=" * 50)
                print(context)
                print("=" * 50)
                
                print("\n2. Mistral's Direct Answer:")
                print("=" * 50)
                print(direct_response)
                print("=" * 50)
                
                print("\n3. Final Response (Combined):")
                print("=" * 50)
                print(combined_response)
                print("=" * 50)
                print()
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                if self.model.config.device == "mps":
                    torch.mps.empty_cache()
                elif self.model.config.device == "cuda":
                    torch.cuda.empty_cache()

def run_model(
    ui_framework: UIFramework,
    deployment_mode: DeploymentMode,
    model_config: Optional[ModelConfig] = None,
    host: str = "0.0.0.0",
    port: Optional[int] = None
) -> None:
    """
    Run the Mistral model with the specified UI framework and deployment mode.
    
    Args:
        ui_framework: UI framework to use
        deployment_mode: Deployment mode
        model_config: Model configuration
        host: Host to bind to
        port: Port to bind to
    """
    # Initialize model
    model_config = model_config or ModelConfig()
    model = MistralModel(model_config)
    model.load()
    
    # Set default port based on framework
    if port is None:
        if ui_framework == UIFramework.STREAMLIT:
            port = 8501
        elif ui_framework == UIFramework.FLASK:
            port = 5000
        elif ui_framework == UIFramework.FASTAPI:
            port = 8000
    
    # Initialize and run UI/API
    if ui_framework == UIFramework.TERMINAL:
        ui = TerminalUI(model)
        ui.run()
    elif ui_framework == UIFramework.STREAMLIT:
        ui = StreamlitUI(model)
        ui.run(host, port)
    elif ui_framework == UIFramework.FLASK:
        api = FlaskAPI(model)
        api.run(host, port)
    elif ui_framework == UIFramework.FASTAPI:
        api = FastAPI(model)
        api.run(host, port)
    else:
        raise ValueError(f"Unsupported UI framework: {ui_framework}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Mistral model")
    parser.add_argument("--ui-framework", type=str,
                      choices=[f.value for f in UIFramework],
                      default=UIFramework.STREAMLIT.value,
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
                      help="Host to bind to")
    parser.add_argument("--port", type=int,
                      help="Port to bind to")
    
    args = parser.parse_args()
    
    # Create model config
    model_config = ModelConfig(model_name=args.model_name)
    
    # Run model
    run_model(
        UIFramework(args.ui_framework),
        DeploymentMode(args.deployment_mode),
        model_config,
        args.host,
        args.port
    )
