#!/usr/bin/env python3
"""
test_chunking.py

Test the improved chunking strategy on a sample of large documents.
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from improved_chunking import ChunkingStrategy, ChunkingConfig, ImprovedChunker

def analyze_chunking_results(chunks, strategy_name):
    """Analyze the results of chunking."""
    # Count chunks
    num_chunks = len(chunks)
    
    # Analyze chunk sizes
    chunk_sizes = [len(chunk['text'].split()) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    
    # Count documents
    doc_ids = set(chunk['doc_id'] for chunk in chunks)
    num_docs = len(doc_ids)
    
    # Count chunks per document
    chunks_per_doc = defaultdict(int)
    for chunk in chunks:
        chunks_per_doc[chunk['doc_id']] += 1
    
    avg_chunks_per_doc = sum(chunks_per_doc.values()) / len(chunks_per_doc) if chunks_per_doc else 0
    
    # Print results
    print(f"\nResults for {strategy_name}:")
    print(f"Total chunks: {num_chunks}")
    print(f"Total documents: {num_docs}")
    print(f"Average chunks per document: {avg_chunks_per_doc:.2f}")
    print(f"Average chunk size: {avg_size:.2f} words")
    print(f"Min chunk size: {min_size} words")
    print(f"Max chunk size: {max_size} words")
    
    # Return statistics for plotting
    return {
        'strategy': strategy_name,
        'num_chunks': num_chunks,
        'num_docs': num_docs,
        'avg_chunks_per_doc': avg_chunks_per_doc,
        'avg_size': avg_size,
        'min_size': min_size,
        'max_size': max_size,
        'chunk_sizes': chunk_sizes
    }

def plot_chunk_size_distribution(results):
    """Plot the distribution of chunk sizes for each strategy."""
    plt.figure(figsize=(12, 8))
    
    for result in results:
        plt.hist(result['chunk_sizes'], bins=20, alpha=0.5, label=result['strategy'])
    
    plt.xlabel('Chunk Size (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Chunk Sizes by Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('chunk_size_distribution.png')
    plt.close()

def test_chunking_strategies(input_dir, sample_size=10):
    """Test different chunking strategies on a sample of documents."""
    # Get all JSON files
    files = list(Path(input_dir).glob('*.json'))
    
    if not files:
        print(f"No JSON files found in {input_dir}")
        return
    
    # Select a random sample of files
    sample_files = random.sample(files, min(sample_size, len(files)))
    
    # Define strategies to test
    strategies = [
        (ChunkingStrategy.FIXED, "Fixed"),
        (ChunkingStrategy.SENTENCE, "Sentence"),
        (ChunkingStrategy.PARAGRAPH, "Paragraph"),
        (ChunkingStrategy.HYBRID, "Hybrid"),
        (ChunkingStrategy.HIERARCHICAL, "Hierarchical")
    ]
    
    # Test each strategy
    all_results = []
    for strategy, name in strategies:
        print(f"\nTesting {name} strategy...")
        
        # Create chunker
        config = ChunkingConfig(strategy=strategy)
        chunker = ImprovedChunker(config)
        
        # Process each file
        all_chunks = []
        for file_path in sample_files:
            try:
                # Read the document
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Extract text to chunk
                text = doc.get('fullText', '')
                if not text:
                    # If no fullText, try to use abstract or other fields
                    text = doc.get('abstract', '')
                    if not text:
                        continue  # Skip documents with no text
                
                # Create chunks
                chunks = chunker.chunk_text(text)
                
                # Add metadata to chunks
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'text': chunk,
                        'source': str(file_path),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'doc_id': doc.get('coreId', doc.get('doi', str(file_path)))
                    }
                    all_chunks.append(chunk_data)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Analyze results
        result = analyze_chunking_results(all_chunks, name)
        all_results.append(result)
    
    # Plot results
    plot_chunk_size_distribution(all_results)
    print("\nChunk size distribution plot saved as 'chunk_size_distribution.png'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test chunking strategies")
    parser.add_argument("--input-dir", type=Path, required=True,
                      help="Directory containing input documents")
    parser.add_argument("--sample-size", type=int, default=10,
                      help="Number of documents to sample")
    
    args = parser.parse_args()
    
    test_chunking_strategies(args.input_dir, args.sample_size) 