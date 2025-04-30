#!/usr/bin/env python3
"""
Memory-efficient script to fix the chunked_data.json file by streaming and processing in batches.
"""

import json
import os
from pathlib import Path
import sys
import re
import time
from tqdm import tqdm
import ijson  # For memory-efficient JSON parsing

def extract_text_from_content(content_str):
    """
    Extract meaningful text from a JSON string in the content field.
    
    Args:
        content_str: The JSON string from the content field
        
    Returns:
        Extracted text as a string
    """
    try:
        # Try to parse as JSON first
        try:
            # Replace improper quotes and fix common JSON errors
            fixed_content = content_str.replace('"\\"', '"\\\"')
            fixed_content = re.sub(r'(\w)"(\w)', r'\1\\"\2', fixed_content)
            
            content_data = json.loads(fixed_content)
            
            # Extract text from the content
            text_parts = []
            
            # Extract title
            if 'title' in content_data and content_data['title']:
                text_parts.append(f"Title: {content_data['title']}")
                
            # Extract authors
            if 'authors' in content_data and content_data['authors']:
                authors = ', '.join(content_data['authors'])
                text_parts.append(f"Authors: {authors}")
                
            # Extract abstract/text field
            for field in ['abstract', 'text', 'content', 'body', 'fullText']:
                if field in content_data and content_data[field]:
                    text_parts.append(f"{field.capitalize()}: {content_data[field]}")
            
            # If we have extracted text, return it
            if text_parts:
                return '\n\n'.join(text_parts)
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing failed, use regex to extract common fields
        title_match = re.search(r'\"title\"\s*\"([^\"]+)\"', content_str)
        if title_match:
            title = title_match.group(1)
            return f"Title: {title}\n\nContent: {content_str}"
        
        # Last resort: just return the raw content
        return content_str
        
    except Exception as e:
        return f"Error extracting text: {str(e)[:100]}"

def process_chunk_batch(input_file, output_file, batch_size=1000, max_chunks=None):
    """
    Process chunks in batches to minimize memory usage.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        batch_size: Number of chunks to process at once
        max_chunks: Maximum number of chunks to process (None for all)
    """
    # Track statistics
    total_chunks = 0
    processed_chunks = 0
    start_time = time.time()
    
    # Open the output file and write the beginning of the array
    with open(output_file, 'w') as out_f:
        out_f.write('[\n')
        
        # Create parser that streams the input
        with open(input_file, 'rb') as in_f:
            # Get an iterator over the chunks in the JSON array
            chunks = ijson.items(in_f, 'item')
            
            # Process chunks in batches
            batch = []
            for i, chunk in enumerate(chunks):
                total_chunks += 1
                
                # Check if we've reached the maximum number of chunks
                if max_chunks is not None and total_chunks > max_chunks:
                    break
                    
                # Process the chunk
                fixed_chunk = chunk.copy()
                
                if 'content' in chunk:
                    # Extract text from content
                    content = chunk['content']
                    text = extract_text_from_content(content)
                    
                    # Add the text field
                    fixed_chunk['text'] = text
                    
                    # Keep content as source but don't include in output to save space
                    fixed_chunk.pop('content', None)
                    
                    processed_chunks += 1
                else:
                    # No content field
                    fixed_chunk['text'] = ''
                
                # Ensure we have proper metadata
                if 'metadata' not in fixed_chunk:
                    fixed_chunk['metadata'] = {}
                
                # Add source and ID to metadata
                fixed_chunk['metadata']['source'] = 'chunked_data'
                fixed_chunk['metadata']['chunk_id'] = chunk.get('chunk_id', '')
                
                # Add to batch
                batch.append(fixed_chunk)
                
                # When batch is full, write to output file
                if len(batch) >= batch_size:
                    batch_json = ',\n'.join(json.dumps(c) for c in batch)
                    if total_chunks > batch_size:  # Add comma if not the first batch
                        out_f.write(',\n')
                    out_f.write(batch_json)
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    chunks_per_sec = total_chunks / elapsed if elapsed > 0 else 0
                    print(f"\rProcessed {total_chunks} chunks ({processed_chunks} with content) - {chunks_per_sec:.2f} chunks/sec", end="")
                    
                    # Clear batch
                    batch = []
            
            # Write any remaining chunks
            if batch:
                batch_json = ',\n'.join(json.dumps(c) for c in batch)
                if total_chunks > len(batch):  # Add comma if not the first batch
                    out_f.write(',\n')
                out_f.write(batch_json)
            
            # Write the end of the array
            out_f.write('\n]')
    
    # Print final statistics
    elapsed = time.time() - start_time
    chunks_per_sec = total_chunks / elapsed if elapsed > 0 else 0
    print(f"\nProcessed {total_chunks} chunks ({processed_chunks} with content) in {elapsed:.2f} seconds")
    print(f"Average processing speed: {chunks_per_sec:.2f} chunks/sec")
    
    return total_chunks, processed_chunks

def main():
    """Main function to fix chunks"""
    # Path to chunks file
    input_file = Path("data/chunks/chunked_data.json")
    output_file = Path("data/chunks/fixed_chunks.json")
    
    # Set batch size (adjust based on available memory)
    batch_size = 5000
    
    # Optional: Set maximum number of chunks to process (for testing)
    max_chunks = None  # Set to None to process all chunks
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    
    print(f"Processing {input_file} in batches of {batch_size}...")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process chunks
    total_chunks, processed_chunks = process_chunk_batch(
        input_file=input_file,
        output_file=output_file,
        batch_size=batch_size,
        max_chunks=max_chunks
    )
    
    print(f"\nFixed chunks saved to {output_file}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks with content: {processed_chunks}")
    print(f"Success rate: {processed_chunks/total_chunks*100:.2f}%")

if __name__ == "__main__":
    # Install ijson if not available
    try:
        import ijson
    except ImportError:
        print("Installing ijson package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ijson"])
        import ijson
    
    main() 