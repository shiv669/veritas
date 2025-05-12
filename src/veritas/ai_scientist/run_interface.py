#!/usr/bin/env python3
"""
Veritas AI Scientist - Research Assistant Interface

This script provides a simple user interface for running different parts of the Veritas AI Scientist,
a research assistant that helps generate research ideas, run experiments, and write up results.

Usage:
    python run_interface.py
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
import subprocess

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/veritas_interface.log")
    ]
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def setup_paths():
    """
    Set up Python paths to include required modules.
    
    Returns:
        Tuple of paths (veritas_path, templates_path)
    """
    veritas_path = PROJECT_ROOT
    templates_path = os.path.join(PROJECT_ROOT, "models", "Cognition")
    
    # Add to Python path if not already there
    if veritas_path not in sys.path:
        sys.path.insert(0, veritas_path)
    if templates_path not in sys.path:
        sys.path.insert(0, templates_path)
    
    logger.info(f"Added to Python path: {veritas_path}, {templates_path}")
    return veritas_path, templates_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Veritas AI Scientist Interface")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "direct", "full", "interactive"],
        default="interactive",
        help="Mode to run: simple (basic test), direct (optimized), "
             "full (comprehensive), or interactive (ask the user)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT_lite",
        help="Research template to use"
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=1,
        help="Number of ideas to generate"
    )
    return parser.parse_args()

def run_simple_test():
    """Run the simple test script."""
    print("Running simple test mode...")
    script_path = os.path.join(os.path.dirname(__file__), "test_simple.py")
    result = subprocess.run([sys.executable, script_path], check=False)
    return result.returncode == 0

def run_direct_implementation(experiment, num_ideas):
    """Run the optimized implementation."""
    print(f"Running optimized mode for {experiment}, generating {num_ideas} idea(s)...")
    script_path = os.path.join(os.path.dirname(__file__), "run_scientist.py")
    result = subprocess.run([
        sys.executable, 
        script_path, 
        "--phase", "idea", 
        "--experiment", experiment, 
        "--num-ideas", str(num_ideas),
        "--use-direct-implementation"
    ], check=False)
    return result.returncode == 0

def run_full_implementation(experiment, num_ideas):
    """Run the comprehensive implementation."""
    print(f"Running comprehensive mode for {experiment}, generating {num_ideas} idea(s)...")
    script_path = os.path.join(os.path.dirname(__file__), "run_scientist.py")
    result = subprocess.run([
        sys.executable, 
        script_path, 
        "--phase", "idea", 
        "--experiment", experiment, 
        "--num-ideas", str(num_ideas)
    ], check=False)
    return result.returncode == 0

def list_available_templates():
    """List available research templates."""
    templates_dir = os.path.join(PROJECT_ROOT, "models", "Cognition", "templates")
    templates = []
    
    try:
        templates = [d for d in os.listdir(templates_dir) 
                    if os.path.isdir(os.path.join(templates_dir, d))]
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        templates = ["nanoGPT_lite"]  # fallback
    
    return templates

def interactive_mode():
    """Interactive mode for the interface."""
    print("=" * 60)
    print("  Veritas AI Scientist - Research Assistant")
    print("=" * 60)
    
    # List available templates
    templates = list_available_templates()
    print("\nAvailable research templates:")
    for i, template in enumerate(templates, 1):
        print(f"  {i}. {template}")
    
    # Select template
    while True:
        try:
            choice = input(f"\nSelect template (1-{len(templates)}) [default: 1]: ")
            if not choice:
                template_idx = 0
                break
            
            template_idx = int(choice) - 1
            if 0 <= template_idx < len(templates):
                break
            else:
                print(f"Please enter a number between 1 and {len(templates)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    experiment = templates[template_idx]
    
    # Select number of ideas
    while True:
        try:
            choice = input("\nNumber of ideas to generate (1-5) [default: 1]: ")
            if not choice:
                num_ideas = 1
                break
            
            num_ideas = int(choice)
            if 1 <= num_ideas <= 5:
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Select mode
    print("\nAvailable modes:")
    print("  1. Simple (fastest)")
    print("  2. Optimized (recommended)")
    print("  3. Comprehensive (in-depth)")
    
    while True:
        try:
            choice = input("\nSelect mode (1-3) [default: 1]: ")
            if not choice:
                mode_idx = 0
                break
            
            mode_idx = int(choice) - 1
            if 0 <= mode_idx < 3:
                break
            else:
                print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    modes = ["simple", "direct", "full"]
    mode = modes[mode_idx]
    
    print("\n" + "=" * 60)
    print(f"Running with:")
    print(f"- Template: {experiment}")
    print(f"- Ideas: {num_ideas}")
    print(f"- Mode: {mode}")
    print("=" * 60)
    
    # Give the user a chance to cancel
    choice = input("\nPress Enter to start, or 'q' to quit: ")
    if choice.lower() == 'q':
        return False
    
    # Run the selected mode
    if mode == "simple":
        return run_simple_test()
    elif mode == "direct":
        return run_direct_implementation(experiment, num_ideas)
    elif mode == "full":
        return run_full_implementation(experiment, num_ideas)

def main():
    """Main function for the interface."""
    try:
        veritas_path, templates_path = setup_paths()
        args = parse_arguments()
        
        # Run in the requested mode
        if args.mode == "interactive":
            success = interactive_mode()
        elif args.mode == "simple":
            success = run_simple_test()
        elif args.mode == "direct":
            success = run_direct_implementation(args.experiment, args.num_ideas)
        elif args.mode == "full":
            success = run_full_implementation(args.experiment, args.num_ideas)
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
        
        # Print result
        if success:
            print("\n✅ Successfully completed!")
            print("Check the logs and results directory for outputs.")
            return 0
        else:
            print("\n❌ Failed to complete operation.")
            print("Check the logs for details.")
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        logger.error(f"Error running Veritas AI Scientist: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 