"""
Utilities for working with Apple's Metal Performance Shaders (MPS) backend
"""
import os
import platform
import torch
from typing import Any, Dict, Optional
from .utils import setup_logging

logger = setup_logging(__name__)

def is_mps_available() -> bool:
    """
    Check if MPS is available on the current system
    
    Returns:
        True if MPS is available, False otherwise
    """
    # Check for macOS and Apple Silicon
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    
    # Check PyTorch MPS availability
    has_mps = hasattr(torch, "backends") and hasattr(torch.backends, "mps")
    mps_available = has_mps and torch.backends.mps.is_available()
    
    return is_mac and is_arm and mps_available


def optimize_for_mps(model: Any) -> Any:
    """
    Optimize a model for MPS
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    if not is_mps_available():
        logger.warning("MPS not available, returning unmodified model")
        return model
    
    # Move model to MPS device
    try:
        model = model.to("mps")
        logger.info("Model successfully moved to MPS device")
    except Exception as e:
        logger.warning(f"Failed to move model to MPS: {e}")
    
    return model


def set_mps_memory_limit(limit_mb: Optional[int] = None) -> None:
    """
    Set MPS memory limit
    
    Args:
        limit_mb: Memory limit in MB, or None to use default
    """
    if not is_mps_available():
        logger.warning("MPS not available, skipping memory limit configuration")
        return
    
    if limit_mb is not None:
        try:
            # Convert MB to bytes
            limit_bytes = limit_mb * 1024 * 1024
            torch.backends.mps.set_cache_memory_limit(limit_bytes)
            logger.info(f"MPS memory limit set to {limit_mb} MB")
        except Exception as e:
            logger.warning(f"Failed to set MPS memory limit: {e}")


def get_mps_memory_stats() -> Dict[str, float]:
    """
    Get MPS memory statistics
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0, 
        "total_mb": 0.0
    }
    
    if not is_mps_available():
        logger.warning("MPS not available, returning empty memory stats")
        return stats
    
    try:
        # Get memory stats (depends on PyTorch version and may not be available)
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
        
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["reserved_mb"] = torch.mps.driver_allocated_memory() / (1024 * 1024)
        
        # Estimate total available memory
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            total_bytes = int(result.stdout.strip())
            stats["total_mb"] = total_bytes / (1024 * 1024)
    except Exception as e:
        logger.warning(f"Failed to get MPS memory stats: {e}")
    
    return stats 