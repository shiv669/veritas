"""
Utility functions for the Veritas RAG system

This module provides various utility functions and classes used throughout the 
Veritas system for common tasks like logging, timing operations, and performance 
measurement.

These utilities improve code organization and reusability by centralizing 
frequently used functionality in a single module.

Features:
- Configurable logging setup
- Performance timing utilities
- Common helper functions

Usage examples:
    from veritas.utils import setup_logging, Timer
    
    # Set up a logger
    logger = setup_logging(__name__)
    logger.info("Starting operation")
    
    # Measure execution time
    with Timer("Document processing"):
        process_documents()
"""
import os
import time
import logging
from typing import Optional
from .config import Config

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with specified name and level.
    
    This function creates a logger with both file and console handlers,
    enabling comprehensive logging across the application with a consistent format.
    
    Args:
        name: The name for the logger, typically __name__ from the calling module
        level: The logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        A configured logger instance
        
    Example:
        logger = setup_logging(__name__)
        logger.info("Application started")
        logger.error("An error occurred")
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
    Context manager and utility class for measuring execution time.
    
    This class provides a simple way to measure and log the execution time
    of code blocks, functions, or specific operations within the application.
    It can be used as a context manager with 'with' statements or directly
    by calling its methods.
    
    Attributes:
        name (str): Descriptive name for what's being timed
        start_time (float): The timestamp when timing began
        logger (Logger): Logger for outputting timing information
        
    Examples:
        # As a context manager
        with Timer("Database query"):
            results = db.execute_query()
            
        # Direct usage
        timer = Timer("Model inference")
        timer.reset()  # Start timing
        model.predict(inputs)
        elapsed = timer.elapsed()
        print(f"Inference took {elapsed:.4f} seconds")
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new timer with optional name.
        
        Args:
            name: Descriptive name for what's being timed
        """
        self.name = name or "Timer"
        self.start_time = None
        self.logger = setup_logging(__name__)
    
    def __enter__(self):
        """Start timing when entering a context block."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log elapsed time when exiting a context block."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name}: {elapsed:.4f} seconds")
    
    def reset(self):
        """Reset the timer to start counting from the current time."""
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """
        Calculate the elapsed time since the timer was started.
        
        Returns:
            Elapsed time in seconds as a float
        """
        return time.time() - self.start_time 