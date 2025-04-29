#!/usr/bin/env python3
"""
Cleanup script for the Veritas repository.
Removes temporary files, logs, and other leftovers.
"""

import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

def run_sudo_command(command):
    """Run a command with sudo if needed."""
    try:
        subprocess.run(['sudo'] + command, check=True)
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to run sudo command: {' '.join(command)}")

def cleanup():
    """Clean up temporary files and directories."""
    # Directories to remove
    dirs_to_remove = [
        '__pycache__',
        '.gradio',
        'test_results',
        'logs',
        'veritas.egg-info',
        '.venv'
    ]
    
    # File patterns to remove
    patterns_to_remove = [
        '._*',  # macOS system files
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.DS_Store',
        '*.log'
    ]
    
    total_items = 0
    # Count total items to process
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            total_items += 1
            
    for pattern in patterns_to_remove:
        total_items += len(list(Path('.').glob(pattern)))
        
    if Path('models').exists():
        for item in Path('models').iterdir():
            if item.is_file() and not item.name.endswith(('.faiss', '.json')):
                total_items += 1
    
    # Initialize progress bar
    with tqdm(total=total_items, desc="Cleaning up files", unit="item") as pbar:
        # Remove directories
        for dir_name in dirs_to_remove:
            if os.path.exists(dir_name):
                try:
                    pbar.set_description(f"Removing directory: {dir_name}")
                    shutil.rmtree(dir_name)
                    pbar.update(1)
                except (PermissionError, OSError) as e:
                    pbar.write(f"Warning: Could not remove {dir_name}: {str(e)}")
        
        # Remove files matching patterns
        for pattern in patterns_to_remove:
            for file in Path('.').glob(pattern):
                try:
                    pbar.set_description(f"Removing file: {file}")
                    file.unlink()
                    pbar.update(1)
                except (PermissionError, OSError) as e:
                    pbar.write(f"Warning: Could not remove {file}: {str(e)}")
        
        # Clean models directory but keep the index files
        models_dir = Path('models')
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_file() and not item.name.endswith(('.faiss', '.json')):
                    try:
                        pbar.set_description(f"Cleaning models: {item}")
                        item.unlink()
                        pbar.update(1)
                    except (PermissionError, OSError) as e:
                        pbar.write(f"Warning: Could not remove {item}: {str(e)}")
    
    print("\nCleanup completed successfully!")

if __name__ == '__main__':
    cleanup() 