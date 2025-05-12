#!/usr/bin/env python3
"""
Type Checking Script for Veritas

This script runs mypy on the Veritas codebase to check for type errors.
It can be used as part of a CI pipeline or for local development.
"""

import os
import sys
import subprocess
import argparse
import tempfile
from typing import List, Optional, Dict, Any

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the modules to check
DEFAULT_MODULES = [
    "src/veritas/typing.py",
    "src/veritas/ai_scientist/test_simple.py",
    "src/veritas/ai_scientist/adapter.py",
    # Add more modules as they get type annotations
]

def run_mypy_with_config(modules: List[str], verbose: bool = False) -> bool:
    """
    Run mypy with a temporary config file to handle import path issues.
    
    Args:
        modules: List of module paths to check
        verbose: Whether to print verbose output
        
    Returns:
        True if all checks pass, False otherwise
    """
    # Create a temporary mypy config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as tmp:
        tmp.write("""
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False
check_untyped_defs = True
disallow_untyped_decorators = False
no_implicit_optional = True
strict_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
namespace_packages = True
explicit_package_bases = True
ignore_missing_imports = True

[mypy.src.veritas.typing]
disallow_untyped_defs = True
disallow_incomplete_defs = True

[mypy.transformers.*]
ignore_missing_imports = True

[mypy.sentence_transformers.*]
ignore_missing_imports = True

[mypy.faiss.*]
ignore_missing_imports = True

[mypy.torch.*]
ignore_missing_imports = False
        """)
        tmp_config_path = tmp.name
    
    try:
        # Set up the mypy command
        cmd = ["mypy", "--config-file", tmp_config_path]
        
        # Add verbose flag if requested
        if verbose:
            cmd.append("--verbose")
        
        # Add the modules to check
        cmd.extend(modules)
        
        # Print the command if verbose
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
        
        # Run mypy
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Return True if mypy exited with 0
        return result.returncode == 0
    except Exception as e:
        print(f"Error running mypy: {e}", file=sys.stderr)
        return False
    finally:
        # Clean up the temporary config file
        try:
            os.unlink(tmp_config_path)
        except:
            pass

def main() -> int:
    """
    Main function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run type checking on Veritas")
    parser.add_argument("--modules", nargs="+", help="Modules to check")
    parser.add_argument("--all", action="store_true", help="Check all Python files")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    # Determine which modules to check
    modules: List[str]
    if args.all:
        # Find all Python files
        modules = []
        for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "src")):
            for file in files:
                if file.endswith(".py"):
                    modules.append(os.path.join(root, file))
        for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "scripts")):
            for file in files:
                if file.endswith(".py"):
                    modules.append(os.path.join(root, file))
    elif args.modules:
        modules = args.modules
    else:
        modules = DEFAULT_MODULES
    
    # Run mypy
    print(f"Checking {len(modules)} modules for type errors...")
    success = run_mypy_with_config(modules, args.verbose)
    
    # Print result
    if success:
        print("✅ All type checks passed!")
        return 0
    else:
        print("❌ Type checking failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 