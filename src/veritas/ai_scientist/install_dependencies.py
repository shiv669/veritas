#!/usr/bin/env python3
"""
Dependency Installation for Veritas AI Scientist

This script helps install the dependencies required for running the Veritas AI Scientist,
a research assistant that helps generate research ideas, conduct experiments, and write papers.
"""

import os
import sys
import subprocess
import argparse
import logging
import importlib.util
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def check_dependency(package):
    """Check if a Python package is installed."""
    try:
        importlib.util.find_spec(package)
        return True
    except ImportError:
        return False

def install_package(package, upgrade=False):
    """Install a Python package using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    
    logger.info(f"Installing {package}...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
        return False

def setup_dependencies(upgrade=False):
    """Install all required dependencies."""
    required_packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "psutil>=5.9.5",
        "pydantic>=2.0.0",
    ]
    
    optional_packages = [
        "rich>=13.0.0",  # For better terminal UI
        "tqdm>=4.65.0",  # For progress bars
    ]
    
    # Install required packages
    success = True
    for package in required_packages:
        pkg_name = package.split(">=")[0]
        if not check_dependency(pkg_name) or upgrade:
            success = install_package(package, upgrade) and success
    
    # Install optional packages
    if success:
        for package in optional_packages:
            pkg_name = package.split(">=")[0]
            if not check_dependency(pkg_name) or upgrade:
                install_package(package, upgrade)
    
    return success

def install_research_dependencies(upgrade=False):
    """Install research template dependencies."""
    # Check if requirements.txt exists
    req_path = os.path.join(PROJECT_ROOT, "models", "Cognition", "requirements.txt")
    if not os.path.exists(req_path):
        logger.error(f"Research template requirements not found at {req_path}")
        return False
    
    # Install dependencies
    logger.info("Installing research template dependencies...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
    if upgrade:
        cmd.append("--upgrade")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install research template dependencies: {e}")
        return False

def main():
    """Main function to install dependencies."""
    parser = argparse.ArgumentParser(description="Install dependencies for Veritas AI Scientist")
    parser.add_argument("--all", action="store_true", help="Install both core and research template dependencies")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade existing packages")
    args = parser.parse_args()
    
    # Check if running in a virtual environment
    if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
        logger.warning("Not running in a virtual environment. It's recommended to use a venv.")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return 1
    
    # Install core dependencies
    success = setup_dependencies(args.upgrade)
    
    # Install research template dependencies if requested
    if args.all:
        success = install_research_dependencies(args.upgrade) and success
    
    if success:
        logger.info("🎉 All dependencies installed successfully!")
        return 0
    else:
        logger.error("❌ Some dependencies failed to install.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 