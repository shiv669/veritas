"""
Utility functions for the Veritas RAG system
"""
import os
import time
import logging
from typing import Optional
from .config import Config

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for a module
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if logger already has handlers to avoid duplicates
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # File handler
        log_file = os.path.join(Config.LOGS_DIR, f"{name.split('.')[-1]}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


class Timer:
    """Simple timer for performance measurement"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize timer
        
        Args:
            name: Optional name for the timer
        """
        self.name = name or "Timer"
        self.start_time = None
        self.logger = setup_logging(__name__)
    
    def __enter__(self):
        """Start timing on entering a context"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log elapsed time on exiting a context"""
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name}: {elapsed:.4f} seconds")
    
    def reset(self):
        """Reset the timer"""
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """
        Get elapsed time since start
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time 