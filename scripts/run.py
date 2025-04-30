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
    GRADIO = "gradio"
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

class GradioUI:
    """Gradio-based user interface."""
    
    def __init__(self, model: MistralModel):
        self.model = model
        # Initialize RAG system
        # Look for timestamped indices (format: YYYYMMDD_HHMMSS)
        index_dirs = [d for d in os.listdir(Config.INDICES_DIR) 
                      if os.path.isdir(os.path.join(Config.INDICES_DIR, d))]
        
        # First try to find timestamped directories
        timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')
        timestamped_dirs = [d for d in index_dirs if timestamp_pattern.match(d)]
        
        if timestamped_dirs:
            # Use the most recent timestamped directory
            latest_index = os.path.join(Config.INDICES_DIR, sorted(timestamped_dirs)[-1])
            logger.info(f"Using timestamped index directory: {latest_index}")
        else:
            # Fall back to any directory if no timestamped ones exist
            latest_index = os.path.join(Config.INDICES_DIR, sorted(index_dirs)[-1])
            logger.info(f"No timestamped index directories found, using: {latest_index}")
        
        from src.veritas.rag import RAGSystem
        self.rag_system = RAGSystem(index_path=latest_index)
    
    def run(self, host: str = "0.0.0.0", port: int = 7860):
        """Run the Gradio interface."""
        try:
            import gradio as gr
        except ImportError:
            logger.error("Gradio is required for this UI framework")
            raise
        
        # Keep a history of conversations
        conversation_history = []
        
        def generate(message, history, top_k):
            # Add user message to history
            history = history or []
            
            # Retrieve relevant context
            retrieved_chunks = self.rag_system.retrieve(message, top_k=top_k)
            
            # Format retrieved context for display
            context_display = ""
            for i, result in enumerate(retrieved_chunks):
                chunk_data = result.get('chunk', {})
                text = chunk_data.get('text', '')[:500]  # Limit context display length
                score = result.get('score', 0)
                context_display += f"<div class='source'><div class='source-header'>Source {i+1} (Score: {score:.4f})</div><div class='source-content'>{text}...</div></div>"
            
            # Construct prompt with context
            context = "\n\n".join([result['chunk'].get('text', '') for result in retrieved_chunks])
            rag_prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {message}

Answer:"""
            
            # Generate answer
            answer = self.model.generate(rag_prompt)
            
            # Return the bot response and updated context
            return answer, context_display
        
        # CSS for styling
        css = """
        :root {
            --primary-color: #10a37f;
            --secondary-color: #f7f7f8;
            --font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            --border-radius: 0.5rem;
        }
        
        body {
            font-family: var(--font-family);
            color: #353740;
            background-color: white;
        }
        
        .container {
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            height: calc(100vh - 120px);
        }
        
        .chat-container {
            flex: 2;
            display: flex;
            flex-direction: column;
            border-radius: var(--border-radius);
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }
        
        .context-container {
            flex: 1;
            margin-left: 20px;
            overflow-y: auto;
            height: 100%;
            border-radius: var(--border-radius);
            background-color: var(--secondary-color);
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
            padding: 1rem;
        }
        
        .source {
            margin-bottom: 15px;
            border: 1px solid #e5e5e5;
            border-radius: var(--border-radius);
            overflow: hidden;
        }
        
        .source-header {
            background-color: #f0f0f0;
            padding: 8px 12px;
            font-weight: 500;
            font-size: 0.9rem;
            border-bottom: 1px solid #e5e5e5;
        }
        
        .source-content {
            padding: 12px;
            font-size: 0.9rem;
            background-color: white;
        }
        
        .title-container {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .title {
            color: var(--primary-color);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: #6e6e80;
        }
        
        .chat-footer {
            border-top: 1px solid #e5e5e5;
            padding: 1rem;
            background-color: white;
        }
        
        .chatbot .user {
            background-color: var(--secondary-color) !important;
        }
        
        .chatbot .assistant {
            background-color: white !important;
        }
        
        .slider-container {
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css) as demo:
            # Header
            with gr.Row(elem_classes=["title-container"]):
                gr.HTML("<h1 class='title'>Veritas RAG</h1>")
                gr.HTML("<p class='subtitle'>Powered by Mistral + FAISS vector search</p>")
            
            # Main content
            with gr.Row(elem_classes=["container"]):
                # Chat column
                with gr.Column(elem_classes=["chat-container"]):
                    chatbot = gr.Chatbot(elem_classes=["chatbot"])
                    
                    with gr.Row(elem_classes=["chat-footer"]):
                        with gr.Column(scale=10):
                            msg = gr.Textbox(
                                placeholder="Ask a question about your documents...", 
                                lines=2,
                                show_label=False
                            )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("Send", variant="primary")
                    
                    with gr.Accordion("Advanced Settings", open=False, elem_classes=["slider-container"]):
                        top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of sources to retrieve")
                
                # Context column
                with gr.Column(elem_classes=["context-container"]):
                    gr.HTML("<h3>Retrieved Sources</h3>")
                    context_display = gr.HTML()
            
            # Define the submit event
            def user_submit(message, history, top_k):
                if not message.strip():
                    return "", history, None
                
                # Add user message to history
                history = history + [[message, None]]
                
                # Return empty message input, updated history, and empty context
                return "", history, None
            
            # Handle the response generation separately
            def respond(message, history, top_k):
                # Get the last user message from history
                last_message = history[-1][0]
                
                # Generate response
                answer, context = generate(last_message, history, top_k)
                
                # Update history with the response
                updated_history = history.copy()
                updated_history[-1][1] = answer
                
                return updated_history, context
            
            # Set up the message submission flow - two steps
            msg.submit(user_submit, [msg, chatbot, top_k], [msg, chatbot, context_display])
            msg.submit(respond, [msg, chatbot, top_k], [chatbot, context_display])
            
            # Same for the button
            submit_btn.click(user_submit, [msg, chatbot, top_k], [msg, chatbot, context_display])
            submit_btn.click(respond, [msg, chatbot, top_k], [chatbot, context_display])
        
        # Launch the app
        demo.launch(server_name=host, server_port=port)

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
        if ui_framework == UIFramework.GRADIO:
            port = 7860
        elif ui_framework == UIFramework.STREAMLIT:
            port = 8501
        elif ui_framework == UIFramework.FLASK:
            port = 5000
        elif ui_framework == UIFramework.FASTAPI:
            port = 8000
    
    # Initialize and run UI/API
    if ui_framework == UIFramework.GRADIO:
        ui = GradioUI(model)
        ui.run(host, port)
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
                      default=UIFramework.GRADIO.value,
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
