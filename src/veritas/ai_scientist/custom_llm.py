"""
Veritas AI Scientist - LLM System Module

This module provides the LLM capabilities for the Veritas AI Scientist system.
It manages model availability and client creation for the research assistant.
"""

import logging
import sys
import importlib.util
from typing import List, Dict, Any, Optional, Union
import os

# Import our adapter
from src.veritas.ai_scientist.adapter import MistralAdapter, create_mistral_client

logger = logging.getLogger(__name__)

# Define our model name
MISTRAL_MODEL_NAME = "mistral-local-rag"

def initialize_llm_system():
    """
    Initialize the LLM system for Veritas AI Scientist.
    
    This function:
    1. Sets up the required LLM components
    2. Registers available models
    3. Configures the client creation system
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # Import scientific modules
        research_templates_path = os.path.join(os.path.dirname(os.getcwd()), "models/Cognition")
        
        # Add templates to path if needed
        if research_templates_path not in sys.path:
            sys.path.append(research_templates_path)
        
        # Import LLM module
        from ai_scientist.llm import AVAILABLE_LLMS, create_client
        
        # Store original functions
        original_create_client = create_client
        
        # Add our model to the list of available models
        if MISTRAL_MODEL_NAME not in AVAILABLE_LLMS:
            AVAILABLE_LLMS.append(MISTRAL_MODEL_NAME)
            logger.info(f"Added {MISTRAL_MODEL_NAME} to available LLMs")
        
        # Create extended create_client function
        def extended_create_client(model):
            """
            Extended version of create_client that supports our Mistral model.
            
            Args:
                model: Model name
                
            Returns:
                Client for the specified model
            """
            if model == MISTRAL_MODEL_NAME:
                logger.info(f"Creating Mistral client for {model}")
                return create_mistral_client()
            else:
                return original_create_client(model)
        
        # Replace the original create_client function
        import ai_scientist.llm
        ai_scientist.llm.create_client = extended_create_client
        
        logger.info("Successfully initialized LLM system")
        return True
    
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure all dependencies are installed correctly")
        return False
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM system: {e}")
        return False

def get_available_llms() -> List[str]:
    """
    Get list of available LLMs for research tasks.
    
    Returns:
        List of available LLM names
    """
    try:
        from ai_scientist.llm import AVAILABLE_LLMS
        return AVAILABLE_LLMS
    except ImportError:
        return [MISTRAL_MODEL_NAME]  # Fallback if modules not imported yet 