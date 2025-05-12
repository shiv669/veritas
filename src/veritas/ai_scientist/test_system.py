#!/usr/bin/env python3
"""
Test script for Veritas AI Scientist with Mistral RAG

This script tests the basic functionality of the system components without
running the full AI Scientist workflow.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def setup_paths():
    """
    Set up Python paths to include both Veritas and AI Scientist.
    
    Returns:
        Tuple of paths (veritas_path, cognition_path)
    """
    # Add the project root to sys.path if not already there
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # Cognition (AI Scientist) path
    cognition_path = os.path.join(PROJECT_ROOT, "models", "Cognition")
    if not os.path.exists(cognition_path):
        logger.error(f"AI Scientist path not found: {cognition_path}")
        raise FileNotFoundError(f"AI Scientist not found at {cognition_path}")
    
    # Add Cognition to sys.path if not already there
    if cognition_path not in sys.path:
        sys.path.append(cognition_path)
    
    logger.info(f"Added to Python path: {PROJECT_ROOT}, {cognition_path}")
    return PROJECT_ROOT, cognition_path

def test_adapter():
    """Test the Mistral adapter"""
    # Test adapter module
    logger.info("\nTesting adapter module...")
    from src.veritas.ai_scientist.adapter import MistralAdapter, create_mistral_client
    
    # Create client
    try:
        client = create_mistral_client("mistral-local-rag")
        logger.info("✅ Successfully created Mistral client")
    except Exception as e:
        logger.error(f"❌ Failed to create Mistral client: {e}")
        success = False
    
    # Test memory manager module
    logger.info("\nTesting memory manager module...")
    from src.veritas.ai_scientist.memory_manager import clean_memory, get_memory_usage, log_memory_usage
    
    try:
        memory_before = get_memory_usage()
        logger.info(f"Memory before cleanup: {memory_before['percent']:.1f}%")
        
        # Run cleanup
        clean_memory()
        
        memory_after = get_memory_usage()
        logger.info(f"Memory after cleanup: {memory_after['percent']:.1f}%")
        logger.info("✅ Successfully tested memory manager")
    except Exception as e:
        logger.error(f"❌ Failed to test memory manager: {e}")
        success = False
    
    # Test custom LLM module
    logger.info("\nTesting LLM module...")
    from src.veritas.ai_scientist.custom_llm import initialize_llm_system, get_available_llms, MISTRAL_MODEL_NAME
    
    try:
        # Try to initialize LLM system
        llm_success = initialize_llm_system()
        if llm_success:
            logger.info(f"✅ Successfully initialized LLM system")
            
            # Check available LLMs
            llms = get_available_llms()
            logger.info(f"Available LLMs: {llms}")
            
            if MISTRAL_MODEL_NAME in llms:
                logger.info(f"✅ {MISTRAL_MODEL_NAME} correctly registered")
            else:
                logger.warning(f"⚠️ {MISTRAL_MODEL_NAME} not found in available LLMs")
        else:
            logger.warning("⚠️ Failed to initialize LLM system (this may be expected)")
    except Exception as e:
        logger.error(f"❌ Failed to test LLM module: {e}")
        success = False
    
    # Test prompt strategy module
    logger.info("\nTesting prompt strategy module...")
    from src.veritas.ai_scientist.prompt_strategy import get_optimized_prompt, format_prompt_for_mistral
    
    try:
        # Get optimized prompts
        idea_prompt = get_optimized_prompt("idea_generation")
        experiment_prompt = get_optimized_prompt("experiments")
        
        # Format a test prompt
        test_prompt = "Please generate some ideas for improving the model."
        formatted_prompt = format_prompt_for_mistral(test_prompt)
        
        logger.info(f"Original prompt: '{test_prompt}'")
        logger.info(f"Formatted prompt: '{formatted_prompt}'")
        
        logger.info("✅ Successfully tested prompt strategy")
    except Exception as e:
        logger.error(f"❌ Failed to test prompt strategy: {e}")
        success = False

def test_memory_manager():
    """Test the memory manager functions"""
    from src.veritas.ai_scientist.memory_manager import clean_memory, get_memory_usage, log_memory_usage
    
    try:
        logger.info("Testing memory manager...")
        mem_before = get_memory_usage()
        logger.info(f"Memory before cleaning: {mem_before['percent']:.1f}%")
        
        clean_memory()
        
        mem_after = get_memory_usage()
        logger.info(f"Memory after cleaning: {mem_after['percent']:.1f}%")
        
        log_memory_usage("test")
        return True
    except Exception as e:
        logger.error(f"Memory manager test failed: {e}")
        return False

def test_custom_llm():
    """Test the custom LLM module"""
    try:
        from src.veritas.ai_scientist.custom_llm import initialize_llm_system, get_available_llms, MISTRAL_MODEL_NAME
        
        logger.info("Testing custom LLM module...")
        logger.info(f"Mistral model name: {MISTRAL_MODEL_NAME}")
        
        try:
            # Only test initialize_llm_system if AI Scientist is available
            logger.info("Testing LLM system initialization...")
            success = initialize_llm_system()
            logger.info(f"Initialization success: {success}")
        except ImportError:
            logger.warning("AI Scientist not available, skipping initialization test")
        
        return True
    except Exception as e:
        logger.error(f"Custom LLM test failed: {e}")
        return False

def test_prompt_strategy():
    """Test the prompt strategy functions"""
    from src.veritas.ai_scientist.prompt_strategy import get_optimized_prompt, format_prompt_for_mistral
    
    try:
        logger.info("Testing prompt strategy...")
        
        # Test optimized prompt
        original_prompt = "You are an AI scientist working on coming up with novel ideas. Your task is to generate creative research directions."
        optimized = get_optimized_prompt("idea_generation", original_prompt)
        logger.info(f"Optimized prompt length: {len(optimized)} chars")
        
        # Test format_prompt_for_mistral
        formatted = format_prompt_for_mistral("Please could you tell me what is the meaning of life?")
        logger.info(f"Formatted prompt: {formatted[:50]}...")
        
        return True
    except Exception as e:
        logger.error(f"Prompt strategy test failed: {e}")
        return False

def run_tests():
    """Run all tests"""
    setup_paths()
    
    # Run tests
    tests = [
        ("Adapter", test_adapter),
        ("Memory Manager", test_memory_manager),
        ("Custom LLM", test_custom_llm),
        ("Prompt Strategy", test_prompt_strategy)
    ]
    
    results = {}
    
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        result = test_func()
        results[name] = result
        logger.info(f"Test {name}: {'SUCCESS' if result else 'FAILURE'}")
    
    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    all_passed = True
    for name, result in results.items():
        logger.info(f"{name}: {'PASS' if result else 'FAIL'}")
        if not result:
            all_passed = False
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    try:
        sys.exit(run_tests())
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1) 