#!/usr/bin/env python3
"""
analyze_chunks.py

Analyze the quality of generated chunks from the processing pipeline.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import random

def load_chunks(chunks_file: str) -> List[Dict[str, Any]]:
    """Load chunks from the JSON file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_chunk_sizes(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of chunk sizes."""
    # Calculate word counts for each chunk
    word_counts = [len(chunk['text'].split()) for chunk in chunks]
    
    # Calculate statistics
    stats = {
        'total_chunks': len(chunks),
        'avg_words': np.mean(word_counts),
        'median_words': np.median(word_counts),
        'min_words': min(word_counts),
        'max_words': max(word_counts),
        'std_dev': np.std(word_counts),
        'percentiles': {
            '25th': np.percentile(word_counts, 25),
            '75th': np.percentile(word_counts, 75),
            '90th': np.percentile(word_counts, 90)
        }
    }
    
    # Count chunks in different size ranges
    size_ranges = {
        '< 100': 0,
        '100-250': 0,
        '250-500': 0,
        '500-750': 0,
        '750-1000': 0,
        '> 1000': 0
    }
    
    for count in word_counts:
        if count < 100:
            size_ranges['< 100'] += 1
        elif count < 250:
            size_ranges['100-250'] += 1
        elif count < 500:
            size_ranges['250-500'] += 1
        elif count < 750:
            size_ranges['500-750'] += 1
        elif count < 1000:
            size_ranges['750-1000'] += 1
        else:
            size_ranges['> 1000'] += 1
    
    stats['size_ranges'] = size_ranges
    
    return stats

def analyze_metadata_completeness(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the completeness of metadata in chunks."""
    metadata_fields = ['title', 'authors', 'abstract', 'year', 'doi', 'coreId']
    completeness = {field: 0 for field in metadata_fields}
    
    for chunk in chunks:
        for field in metadata_fields:
            if field in chunk and chunk[field] is not None:
                completeness[field] += 1
    
    # Convert to percentages
    total_chunks = len(chunks)
    completeness = {k: (v / total_chunks) * 100 for k, v in completeness.items()}
    
    return completeness

def analyze_document_coverage(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how many chunks are generated per document."""
    chunks_per_doc = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk['doc_id']
        chunks_per_doc[doc_id] += 1
    
    stats = {
        'total_documents': len(chunks_per_doc),
        'avg_chunks_per_doc': np.mean(list(chunks_per_doc.values())),
        'median_chunks_per_doc': np.median(list(chunks_per_doc.values())),
        'min_chunks_per_doc': min(chunks_per_doc.values()),
        'max_chunks_per_doc': max(chunks_per_doc.values())
    }
    
    return stats

def sample_chunks_for_quality_check(chunks: List[Dict[str, Any]], num_samples: int = 5) -> List[Dict[str, Any]]:
    """Get a random sample of chunks for manual quality inspection."""
    return random.sample(chunks, min(num_samples, len(chunks)))

def plot_chunk_size_distribution(chunks: List[Dict[str, Any]], output_file: str = 'chunk_size_distribution.png'):
    """Plot the distribution of chunk sizes."""
    word_counts = [len(chunk['text'].split()) for chunk in chunks]
    
    plt.figure(figsize=(12, 6))
    plt.hist(word_counts, bins=50, edgecolor='black')
    plt.title('Distribution of Chunk Sizes')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    plt.close()

def main():
    chunks_file = "data/chunks.json"
    chunks = load_chunks(chunks_file)
    
    print("\n=== Chunk Size Analysis ===")
    size_stats = analyze_chunk_sizes(chunks)
    print(f"Total chunks: {size_stats['total_chunks']}")
    print(f"Average words per chunk: {size_stats['avg_words']:.2f}")
    print(f"Median words per chunk: {size_stats['median_words']:.2f}")
    print(f"Min words: {size_stats['min_words']}")
    print(f"Max words: {size_stats['max_words']}")
    print("\nSize distribution:")
    for range_name, count in size_stats['size_ranges'].items():
        percentage = (count / size_stats['total_chunks']) * 100
        print(f"{range_name}: {count} chunks ({percentage:.1f}%)")
    
    print("\n=== Metadata Completeness ===")
    metadata_stats = analyze_metadata_completeness(chunks)
    for field, percentage in metadata_stats.items():
        print(f"{field}: {percentage:.1f}% complete")
    
    print("\n=== Document Coverage ===")
    coverage_stats = analyze_document_coverage(chunks)
    print(f"Total documents: {coverage_stats['total_documents']}")
    print(f"Average chunks per document: {coverage_stats['avg_chunks_per_doc']:.2f}")
    print(f"Median chunks per document: {coverage_stats['median_chunks_per_doc']:.2f}")
    print(f"Min chunks per document: {coverage_stats['min_chunks_per_doc']}")
    print(f"Max chunks per document: {coverage_stats['max_chunks_per_doc']}")
    
    # Generate and save size distribution plot
    plot_chunk_size_distribution(chunks)
    print("\nChunk size distribution plot saved as 'chunk_size_distribution.png'")
    
    # Sample chunks for manual inspection
    print("\n=== Sample Chunks for Quality Check ===")
    samples = sample_chunks_for_quality_check(chunks)
    for i, chunk in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"Document: {chunk.get('title', 'No title')}")
        print(f"Chunk {chunk['chunk_index'] + 1} of {chunk['total_chunks']}")
        print(f"Words: {len(chunk['text'].split())}")
        print("First 200 characters of text:")
        print(chunk['text'][:200] + "...")

if __name__ == "__main__":
    main() 