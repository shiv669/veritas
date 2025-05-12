#!/usr/bin/env python3
"""
Unified Installation Script for Veritas

This script helps install all components of the Veritas system including:
- The core RAG system
- The AI Scientist component
- All dependencies
- Model downloads (optional)
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def install_dependencies(upgrade=False, ignore_errors=False):
    """Install dependencies from requirements.txt."""
    logger.info("Installing dependencies from requirements.txt...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    
    if upgrade:
        cmd.append("--upgrade")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install all dependencies: {e}")
        
        if ignore_errors:
            logger.warning("Continuing despite dependency installation errors")
            return True
        return False

def install_package(ignore_errors=False):
    """Install the Veritas package itself."""
    logger.info("Installing Veritas package...")
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Veritas package installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install package: {e}")
        
        if ignore_errors:
            logger.warning("Continuing despite package installation error")
            return True
        return False

def create_directories():
    """Create necessary directories for the system."""
    dirs = [
        "models/mistral",
        "models/faiss",
        "models/Cognition/templates",
        "data/input",
        "data/output",
        "data/indices/latest",
        "logs",
        "tmp"
    ]
    
    logger.info("Creating necessary directories...")
    for directory in dirs:
        path = os.path.join(PROJECT_ROOT, directory)
        os.makedirs(path, exist_ok=True)
    
    logger.info("✅ Directories created successfully")
    return True

def download_model(model_name=None, ignore_errors=False):
    """Download the Mistral model."""
    if not model_name:
        model_name = "mistralai/Mistral-7B-v0.2"
    
    logger.info(f"Downloading model {model_name}...")
    model_dir = os.path.join(PROJECT_ROOT, "models/mistral")
    
    try:
        # First try to import huggingface_hub
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.warning("huggingface_hub not installed. Attempting to install it...")
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            from huggingface_hub import snapshot_download
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir
        )
        
        logger.info(f"✅ Model {model_name} downloaded successfully to {model_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        
        if ignore_errors:
            logger.warning("Continuing despite model download error")
            return True
        return False

def download_research_templates(ignore_errors=False):
    """Download research templates for AI Scientist."""
    logger.info("Setting up research templates...")
    templates_dir = os.path.join(PROJECT_ROOT, "models/Cognition/templates")
    
    try:
        # For now, we'll simply create a basic template directory structure
        # In a real implementation, this would download templates from a source
        example_template = os.path.join(templates_dir, "nanoGPT_lite")
        os.makedirs(example_template, exist_ok=True)
        
        # Create a readme file for the template
        readme_path = os.path.join(example_template, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write("# nanoGPT Lite Research Template\n\nThis is a template for AI research on lightweight neural network architectures.")
        
        logger.info("✅ Research templates set up successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to set up research templates: {e}")
        
        if ignore_errors:
            logger.warning("Continuing despite template setup error")
            return True
        return False

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install Veritas system")
    parser.add_argument("--download-model", action="store_true", help="Download the Mistral model")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.2", help="Model to download")
    parser.add_argument("--skip-dependencies", action="store_true", help="Skip installing dependencies")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade existing packages")
    parser.add_argument("--ignore-errors", action="store_true", help="Continue installation even if some steps fail")
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*70)
    print("  VERITAS INSTALLATION - UNIFIED SETUP")
    print("="*70 + "\n")
    
    # Check Python version
    if sys.version_info < (3, 9):
        logger.error("❌ Python 3.9 or higher is required")
        return 1
    
    # Check if running in a virtual environment
    if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
        logger.warning("⚠️ Not running in a virtual environment. It's recommended to use a venv.")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return 1
    
    success = True
    
    # Create directories
    if not create_directories():
        if not args.ignore_errors:
            return 1
        success = False
    
    # Install dependencies
    if not args.skip_dependencies and not install_dependencies(upgrade=args.upgrade, ignore_errors=args.ignore_errors):
        if not args.ignore_errors:
            return 1
        success = False
    
    # Install package
    if not install_package(ignore_errors=args.ignore_errors):
        if not args.ignore_errors:
            return 1
        success = False
    
    # Set up research templates
    if not download_research_templates(ignore_errors=args.ignore_errors):
        if not args.ignore_errors:
            logger.warning("⚠️ Failed to set up research templates")
        success = False
    
    # Download model if requested
    if args.download_model:
        if not download_model(args.model, ignore_errors=args.ignore_errors):
            if not args.ignore_errors:
                return 1
            success = False
    
    if success:
        print("\n" + "="*70)
        print("  VERITAS INSTALLATION COMPLETE")
        print("="*70)
        print("\nTo get started, run:\n")
        print("  python scripts/run.py")
        print("\nFor AI Scientist, run:\n")
        print("  python scripts/run.py --system ai_scientist")
        print("\nOr use the command-line tools:\n")
        print("  veritas")
        print("  veritas-ai-scientist")
        print("\nSee the documentation for more information.")
        return 0
    else:
        if args.ignore_errors:
            logger.warning("⚠️ Installation completed with some warnings or errors")
            logger.warning("   The system may not function correctly. Check the logs for details.")
            return 0
        else:
            logger.error("❌ Installation completed with errors")
            return 1

if __name__ == "__main__":
    sys.exit(main()) 