"""
MPS (Metal Performance Shaders) utilities for Apple Silicon

What this file does:
This file makes Veritas run faster on Apple's M-series chips (M1, M2, M3, M4).
It's like a special set of instructions that lets the AI take advantage of
the powerful graphics capabilities in Apple Silicon computers.

Without this, the system would still work but would run much slower.
"""
import torch
import logging
import os
from typing import Any, Dict, Optional
from .config import Config

logger = logging.getLogger(__name__)

def is_mps_available() -> bool:
    """
    Checks if your Mac has Apple Silicon and can use these special optimizations
    
    Returns:
    - True if you have an M-series Mac that can use these speed boosts
    - False if you have an older Mac or a PC
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
    Controls how much memory the AI can use on your Mac
    
    This is like telling the AI how much of your computer's RAM it's allowed to use.
    Setting this properly prevents crashes and out-of-memory errors.
    
    Parameters:
    - memory_limit_mb: Maximum RAM the AI can use (in megabytes)
    
    Returns:
    - True if the setting was applied successfully
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
    Apply MPS-specific optimizations to an object
    
    This function handles runtime operations like converting models to
    half precision and moving them to the MPS device. These are operations
    that need to happen during execution, not just configuration settings.
    
    Args:
        obj: Object to optimize (typically a model)
        
    Returns:
        Optimized object
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
    Clear MPS memory cache to free up resources
    
    This is a runtime operation that should be called when memory
    usage gets high during model execution.
    """
    if is_mps_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache")
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")

def optimize_memory_for_m4() -> None:
    """
    Apply memory optimizations for M4 Mac
    
    This function coordinates both environment settings (via Config)
    and runtime operations for optimal M4 performance.
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