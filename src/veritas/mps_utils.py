"""
MPS (Metal Performance Shaders) utilities for Apple Silicon
"""
import torch
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def is_mps_available() -> bool:
    """
    Check if MPS is available on this system.
    
    Returns:
        True if MPS is available, False otherwise
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
    Set MPS memory limit in MB.
    
    Args:
        memory_limit_mb: Memory limit in MB
        
    Returns:
        True if successful, False otherwise
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
            # Try environment variable as fallback
            memory_limit_gb = memory_limit_mb / 1024
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable automatic growth
            os.environ["PYTORCH_MPS_MEMORY_LIMIT"] = f"{memory_limit_gb}GB"
            logger.info(f"Set MPS memory limit via environment variables: {memory_limit_gb}GB")
            return True
    except Exception as e:
        logger.warning(f"Failed to set MPS memory limit: {e}")
        return False

def optimize_for_mps(obj: Any) -> Any:
    """
    Apply MPS-specific optimizations to an object
    
    Args:
        obj: Object to optimize
        
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

def optimize_memory_for_m4() -> None:
    """
    Apply memory optimizations for M4 Mac
    """
    # Set PyTorch memory management options
    if is_mps_available():
        # Disable automatic growth
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Set MPS memory limit to 80% of system RAM
        os.environ["PYTORCH_MPS_MEMORY_LIMIT"] = "102GB"  # 80% of 128GB
        
        # Other Apple-specific optimizations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        logger.info("Applied memory optimizations for M4 Mac") 