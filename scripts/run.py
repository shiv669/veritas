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

class UIFramework(Enum):
    """Supported UI frameworks."""
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
    max_new_tokens: int = 1024  # Default max tokens to generate
    temperature: float = 0.7    # Default temperature
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    device: str = get_device()

class MistralModel:
    """Wrapper for Mistral model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
        ).to(self.config.device)
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    if ui_framework == UIFramework.STREAMLIT:
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
