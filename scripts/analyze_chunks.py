#!/usr/bin/env python3
"""
Script to analyze chunk data and generate statistics.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Any
from pathlib import Path

# Add the project root to Python path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.veritas.config import Config
from src.veritas.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)

def load_chunks(chunks_file: str) -> List[Dict[str, Any]]:
    """Load chunks from JSON file."""
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chunks file: {e}")
        return []

def analyze_chunk_sizes(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze chunk sizes and generate statistics."""
    chunk_sizes = [len(chunk.get("content", "")) for chunk in chunks]
    
    return {
        "count": len(chunk_sizes),
        "min_size": min(chunk_sizes) if chunk_sizes else 0,
        "max_size": max(chunk_sizes) if chunk_sizes else 0,
        "avg_size": np.mean(chunk_sizes) if chunk_sizes else 0,
        "median_size": np.median(chunk_sizes) if chunk_sizes else 0,
        "std_dev": np.std(chunk_sizes) if chunk_sizes else 0,
        "size_distribution": Counter([round(size, -2) for size in chunk_sizes])
    }

def plot_size_distribution(stats: Dict[str, Any], output_file: str):
    """Plot chunk size distribution."""
    plt.figure(figsize=(12, 6))
    
    # Convert Counter to sorted lists for plotting
    sizes = sorted(stats["size_distribution"].keys())
    counts = [stats["size_distribution"][size] for size in sizes]
    
    plt.bar(sizes, counts, width=50)
    plt.xlabel("Chunk Size (characters)")
    plt.ylabel("Number of Chunks")
    plt.title("Chunk Size Distribution")
    
    # Add stats as text
    stats_text = (
        f"Total Chunks: {stats['count']}\n"
        f"Min Size: {stats['min_size']:.1f}\n"
        f"Max Size: {stats['max_size']:.1f}\n"
        f"Average Size: {stats['avg_size']:.1f}\n"
        f"Median Size: {stats['median_size']:.1f}\n"
        f"Std Dev: {stats['std_dev']:.1f}"
    )
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords="axes fraction",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def main():
    """Main function to analyze chunks."""
    # Define file paths using Config
    chunks_file = os.path.join(Config.CHUNKS_DIR, "chunked_data.json")
    output_dir = Config.OUTPUT_DIR
    output_file = os.path.join(Config.DOCS_DIR, "chunk_size_distribution.png")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and analyze chunks
    logger.info(f"Loading chunks from {chunks_file}")
    chunks = load_chunks(chunks_file)
    
    if not chunks:
        logger.error("No chunks found for analysis")
        return
    
    logger.info(f"Analyzing {len(chunks)} chunks")
    stats = analyze_chunk_sizes(chunks)
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, "chunk_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        # Convert Counter to dict for JSON serialization
        stats_json = stats.copy()
        stats_json["size_distribution"] = dict(stats["size_distribution"])
        json.dump(stats_json, f, indent=2)
    
    # Plot size distribution
    logger.info(f"Plotting chunk size distribution to {output_file}")
    plot_size_distribution(stats, output_file)
    
    logger.info("Analysis complete")
    logger.info(f"Statistics saved to {stats_file}")
    logger.info(f"Distribution plot saved to {output_file}")

if __name__ == "__main__":
    main() 