#!/usr/bin/env python3
"""
mps_utils.py

Utility functions for optimizing performance with Metal Performance Shaders (MPS) on macOS.
"""

import torch
import logging
import os
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

def get_optimal_device(task_type: str = "inference") -> str:
    """
    Determines the best device to use based on availability and task type.
    
    Args:
        task_type (str): Type of task. Can be "inference" or "embedding".
                        Defaults to "inference".
    
    Returns:
        str: The optimal device to use ('mps', 'cuda', or 'cpu')
    """
    if task_type not in ["inference", "embedding"]:
        raise ValueError("task_type must be either 'inference' or 'embedding'")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        return "cuda"  # CUDA is always preferred when available
    
    # Check if MPS is available
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        if task_type == "inference":
            return "mps"  # MPS is better for inference
        else:
            return "cpu"  # CPU is better for embeddings
    
    # Fallback to CPU
    return "cpu"

def optimize_for_mps(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize a PyTorch model for MPS device.
    
    Args:
        model: The PyTorch model to optimize
        
    Returns:
        The optimized model
    """
    if not torch.backends.mps.is_available():
        logger.warning("MPS not available, skipping optimization")
        return model
    
    # Convert model to float32 for better MPS compatibility
    model = model.float()
    
    # Move model to MPS device
    model = model.to("mps")
    
    logger.info("Model optimized for MPS")
    return model

def get_optimal_batch_size(device: str, base_batch_size: int = 32) -> int:
    """
    Determine the optimal batch size for the given device.
    
    Args:
        device: The device to optimize for
        base_batch_size: The base batch size to use
        
    Returns:
        int: The optimized batch size
    """
    if device == "mps":
        # MPS works better with smaller batch sizes
        return min(base_batch_size, 32)
    elif device == "cuda":
        # CUDA can handle larger batch sizes
        return base_batch_size * 2
    else:
        # CPU performance varies based on available memory
        return base_batch_size

def prepare_inputs_for_mps(
    inputs: Dict[str, torch.Tensor],
    device: str = "mps"
) -> Dict[str, torch.Tensor]:
    """
    Prepare model inputs for MPS device.
    
    Args:
        inputs: The model inputs
        device: The device to prepare for
        
    Returns:
        Dict[str, torch.Tensor]: The prepared inputs
    """
    if device != "mps":
        return inputs
    
    # Ensure inputs are on the correct device
    prepared_inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Ensure attention mask is set
    if "attention_mask" not in prepared_inputs and "input_ids" in prepared_inputs:
        prepared_inputs["attention_mask"] = torch.ones_like(prepared_inputs["input_ids"])
    
    return prepared_inputs

def get_memory_info() -> Dict[str, Any]:
    """
    Get memory information for the current device.
    
    Returns:
        Dict[str, Any]: Memory information
    """
    info = {}
    
    if torch.backends.mps.is_available():
        # MPS doesn't provide direct memory info, but we can log device info
        info["device"] = "mps"
        info["available"] = torch.backends.mps.is_available()
        info["built"] = torch.backends.mps.is_built()
    elif torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated"] = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        info["memory_reserved"] = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
    else:
        info["device"] = "cpu"
        import psutil
        info["memory_available"] = psutil.virtual_memory().available / (1024 ** 3)  # GB
        info["memory_total"] = psutil.virtual_memory().total / (1024 ** 3)  # GB
    
    return info

def log_memory_info():
    """Log memory information for the current device."""
    info = get_memory_info()
    logger.info(f"Memory info: {info}") 