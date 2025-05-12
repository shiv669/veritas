"""
Veritas AI Scientist - Memory Management Module

This module provides functions to manage memory usage during scientific research workflows,
which can be memory-intensive due to multiple LLM calls and experiment processing.
Optimized for Apple Silicon M-series chips.
"""

import gc
import logging
import os
import psutil
import torch
from src.veritas.mps_utils import clear_mps_cache, is_mps_available

logger = logging.getLogger(__name__)

def get_memory_usage():
    """
    Get current memory usage statistics.
    
    Returns:
        Dict with memory usage information
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / (1024 * 1024),  # RSS in MB
        "vms": memory_info.vms / (1024 * 1024),  # VMS in MB
        "percent": process.memory_percent(),
        "available": psutil.virtual_memory().available / (1024 * 1024)  # Available in MB
    }

def log_memory_usage(stage=""):
    """
    Log current memory usage.
    
    Args:
        stage: Name of the current stage for logging
    """
    memory = get_memory_usage()
    logger.info(f"Memory usage {stage}: RSS={memory['rss']:.1f}MB, "
                f"Available={memory['available']:.1f}MB, "
                f"Percent={memory['percent']:.1f}%")
    
def clean_memory(force_gpu=False):
    """
    Clean up memory to prevent OOM errors.
    
    Args:
        force_gpu: Whether to force GPU cache cleanup even if not at threshold
    """
    # Run Python garbage collection
    gc.collect()
    
    # Clear MPS cache if on Apple Silicon
    if is_mps_available():
        clear_mps_cache()
    
    # Log memory usage after cleanup
    log_memory_usage("after cleanup")

def memory_check_decorator(threshold_percent=80):
    """
    Decorator to check memory usage before and after function execution.
    
    Args:
        threshold_percent: Percentage threshold to trigger memory cleanup
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check memory before
            pre_memory = get_memory_usage()
            logger.info(f"Running {func.__name__}: pre-memory={pre_memory['percent']:.1f}%")
            
            # Run the function
            result = func(*args, **kwargs)
            
            # Check memory after
            post_memory = get_memory_usage()
            logger.info(f"Completed {func.__name__}: post-memory={post_memory['percent']:.1f}%")
            
            # Clean if threshold exceeded
            if post_memory['percent'] > threshold_percent:
                logger.warning(f"Memory usage high ({post_memory['percent']:.1f}%), cleaning")
                clean_memory(force_gpu=True)
            
            return result
        return wrapper
    return decorator

def wrap_ai_scientist_functions():
    """
    Apply memory management to key scientific functions.
    
    This function enhances memory-intensive research functions with
    active memory management to prevent OOM errors during long-running tasks.
    """
    try:
        from ai_scientist.generate_ideas import generate_ideas
        from ai_scientist.perform_experiments import perform_experiments
        from ai_scientist.perform_writeup import perform_writeup
        
        # Store original functions
        original_generate_ideas = generate_ideas
        original_perform_experiments = perform_experiments
        original_perform_writeup = perform_writeup
        
        # Apply memory decorator
        generate_ideas_decorated = memory_check_decorator()(original_generate_ideas)
        perform_experiments_decorated = memory_check_decorator()(original_perform_experiments)
        perform_writeup_decorated = memory_check_decorator()(original_perform_writeup)
        
        # Replace functions with decorated versions
        import ai_scientist.generate_ideas
        import ai_scientist.perform_experiments
        import ai_scientist.perform_writeup
        
        ai_scientist.generate_ideas.generate_ideas = generate_ideas_decorated
        ai_scientist.perform_experiments.perform_experiments = perform_experiments_decorated
        ai_scientist.perform_writeup.perform_writeup = perform_writeup_decorated
        
        logger.info("Successfully enhanced scientific functions with memory management")
        return True
    except Exception as e:
        logger.error(f"Failed to enhance functions with memory management: {e}")
        return False 