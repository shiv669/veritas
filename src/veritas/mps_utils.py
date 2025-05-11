"""
MPS (Metal Performance Shaders) utilities for Apple Silicon

This module provides specialized optimization functions for running the Veritas
RAG system on Apple Silicon hardware (M1, M2, M3, M4 chips). It improves performance
by leveraging Metal Performance Shaders (MPS) and implementing Mac-specific
memory management strategies.

Features:
- MPS availability detection
- Memory management optimizations
- Cache clearing utilities
- Model optimization for Apple Silicon
- M4 Mac-specific configurations

Usage examples:
    from veritas.mps_utils import is_mps_available, clear_mps_cache, optimize_for_mps
    
    # Check if MPS is available
    if is_mps_available():
        # Apply optimizations to model
        model = optimize_for_mps(model)
        
    # Clear cache after heavy operations
    clear_mps_cache()
"""
import torch
import logging
import os
import gc
from typing import Any, Dict, Optional
from .config import Config

logger = logging.getLogger(__name__)

def is_mps_available() -> bool:
    """
    Check if Metal Performance Shaders (MPS) are available on the current device.
    
    This function verifies that the current system supports Apple's MPS backend
    for PyTorch by checking the torch.backends.mps availability and attempting
    to create a test tensor on the MPS device.
    
    Returns:
        bool: True if MPS is available and functional, False otherwise
        
    Example:
        if is_mps_available():
            device = "mps"
        else:
            device = "cpu"
    """
    try:
        if not torch.backends.mps.is_available():
            return False
            
        # Test MPS by creating a small tensor
        test_tensor = torch.zeros(1).to("mps")
        del test_tensor
        return True
    except Exception as e:
        logger.warning(f"MPS availability check failed: {e}")
        return False

def set_mps_memory_limit(memory_limit_mb: int) -> bool:
    """
    Set the memory limit for MPS (Metal Performance Shaders) operations.
    
    This function controls the maximum amount of memory that PyTorch can allocate
    on the Apple Silicon GPU. It uses either the native PyTorch API or environment
    variables depending on the PyTorch version.
    
    Args:
        memory_limit_mb: Maximum memory in megabytes to allocate for MPS operations
        
    Returns:
        bool: True if the limit was successfully set, False otherwise
        
    Example:
        # Limit MPS memory to 16GB
        set_mps_memory_limit(16 * 1024)
    """
    try:
        # Check if torch version supports this feature
        if hasattr(torch.mps, 'set_per_process_memory_fraction'):
            # Convert MB to fraction of total memory
            total_memory = 128 * 1024  # Assume 128GB for M4
            fraction = min(memory_limit_mb / total_memory, 0.8)  # Cap at 80%
            torch.mps.set_per_process_memory_fraction(fraction)
            logger.info(f"Set MPS memory fraction to {fraction:.2f}")
            return True
        else:
            # Use centralized config settings
            Config.MEMORY_LIMIT_GB = memory_limit_mb / 1024
            Config.setup_environment()  # Apply settings through centralized method
            logger.info(f"Set MPS memory limit via environment variables: {Config.MEMORY_LIMIT_GB}GB")
            return True
    except Exception as e:
        logger.warning(f"Failed to set MPS memory limit: {e}")
        return False

def optimize_for_mps(obj: Any) -> Any:
    """
    Apply Apple Silicon-specific optimizations to a model or other PyTorch object.
    
    This function applies various optimizations to make models run efficiently on 
    Apple Silicon hardware, including converting to half precision and moving tensors 
    to the MPS device.
    
    Args:
        obj: The object to optimize (typically a model instance or RAG system)
        
    Returns:
        The optimized object with MPS-specific enhancements
        
    Example:
        model = AutoModelForCausalLM.from_pretrained("mistral-7b")
        model = optimize_for_mps(model)
    """
    try:
        if hasattr(obj, 'model') and hasattr(obj.model, 'half'):
            # Use half precision for model if possible
            try:
                obj.model = obj.model.half()
                logger.info("Converted model to half precision")
            except Exception as e:
                logger.warning(f"Failed to convert model to half precision: {e}")
                
        if hasattr(obj, 'model') and hasattr(obj.model, 'to'):
            # Ensure model is on MPS device
            try:
                obj.model = obj.model.to("mps")
                logger.info("Moved model to MPS device")
            except Exception as e:
                logger.warning(f"Failed to move model to MPS device: {e}")
                
        if hasattr(obj, 'embedding_model') and hasattr(obj.embedding_model, 'to'):
            # Ensure embedding model is on MPS device
            try:
                obj.embedding_model.to("mps")
                logger.info("Moved embedding model to MPS device")
            except Exception as e:
                logger.warning(f"Failed to move embedding model to MPS device: {e}")
                
        return obj
    except Exception as e:
        logger.warning(f"Failed to optimize for MPS: {e}")
        return obj

def clear_mps_cache() -> None:
    """
    Clear the MPS device memory cache to free up GPU resources.
    
    This function should be called periodically during processing to prevent
    memory buildup on the Apple Silicon GPU, especially after large operations
    or when memory warnings occur.
    
    Example:
        # After generating text with a large model
        result = model.generate(inputs)
        clear_mps_cache()  # Free up GPU memory
    """
    if is_mps_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache")
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")

def clear_memory() -> None:
    """
    Perform comprehensive memory cleanup across both CPU and GPU resources.
    
    This function combines Python garbage collection with MPS-specific cleanup
    to maximize available memory after resource-intensive operations.
    
    Example:
        # After completing a batch of operations
        process_large_documents()
        clear_memory()  # Thorough cleanup of all resources
    """
    # Always collect garbage first
    gc.collect()
    
    # Clear MPS cache if available
    if is_mps_available():
        clear_mps_cache()
    
    # More aggressive tensor cleanup
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
    
    # Final garbage collection
    gc.collect()
    logger.info("Aggressively cleared memory")

def optimize_memory_for_m4() -> None:
    """
    Apply comprehensive memory optimizations specifically tuned for M4 Macs.
    
    This function configures both environment variables and runtime settings
    to maximize performance on M4 Mac hardware with 128GB RAM. It should be
    called at application startup.
    
    Example:
        # At the beginning of your application
        optimize_memory_for_m4()
        load_models()
    """
    # Apply environment variable settings using centralized Config
    if is_mps_available():
        # Update Config with M4-specific settings
        Config.MEMORY_LIMIT_GB = 102  # 80% of 128GB
        Config.CPU_THREADS = 4  # Optimal thread count for M4
        
        # Apply the updated settings
        Config.setup_environment()
        
        # Also perform runtime cleanup if needed
        clear_mps_cache()
        
        logger.info("Applied memory optimizations for M4 Mac") 