"""
Utility functions for the Veritas RAG system

What this file does:
This file contains helpful tools used throughout the Veritas system.
Think of it like a toolbox with various useful gadgets that other parts
of the system can use when needed.
"""
import os
import time
import logging
from typing import Optional
from .config import Config

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates a system for recording what happens while Veritas runs
    
    This is like setting up a diary that Veritas writes in to keep track
    of what it's doing. If something goes wrong, you can read this diary
    to figure out what happened.
    
    Parameters:
    - name: What part of the system is writing the log
    - level: How detailed the log should be
    
    Returns:
    - A configured logger ready to record information
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
    """
    Keeps track of how long things take to run
    
    This is like a stopwatch that helps measure performance and
    identify parts of the system that might be running slowly.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Sets up a new timer
        
        Parameters:
        - name: What you're timing (e.g., "Loading Model" or "Searching Documents")
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