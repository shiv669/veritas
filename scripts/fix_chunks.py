#!/usr/bin/env python3
"""
Script to fix the chunked_data.json file by extracting proper text content.
"""

import json
import os
from pathlib import Path
import sys
import re
from tqdm import tqdm

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
        print(f"Error extracting text: {e}")
        return content_str

def main():
    """Main function to fix chunks"""
    # Path to chunks file
    chunks_file = Path("data/chunks/chunked_data.json")
    output_file = Path("data/chunks/fixed_chunks.json")
    
    if not chunks_file.exists():
        print(f"Chunks file not found: {chunks_file}")
        return
    
    print(f"Processing {chunks_file}...")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all chunks
    with open(chunks_file, 'r') as f:
        try:
            chunks = json.load(f)
            print(f"Loaded {len(chunks)} chunks")
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")
            return
    
    # Sample a few chunks to test extraction
    sample_size = min(5, len(chunks))
    samples = []
    
    for i in range(sample_size):
        chunk = chunks[i]
        if 'content' in chunk:
            # Extract text from content
            raw_content = chunk['content']
            extracted_text = extract_text_from_content(raw_content)
            
            samples.append({
                'chunk_id': chunk.get('chunk_id', f'sample_{i}'),
                'raw_content': raw_content[:100] + '...' if len(raw_content) > 100 else raw_content,
                'extracted_text': extracted_text[:100] + '...' if len(extracted_text) > 100 else extracted_text
            })
    
    # Print samples
    print("\nSample extraction results:")
    for sample in samples:
        print(f"\nChunk ID: {sample['chunk_id']}")
        print(f"Raw content: {sample['raw_content']}")
        print(f"Extracted text: {sample['extracted_text']}")
    
    # Ask for confirmation
    confirm = input("\nProcess all chunks? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Process all chunks
    fixed_chunks = []
    print(f"Processing {len(chunks)} chunks...")
    
    for chunk in tqdm(chunks):
        fixed_chunk = chunk.copy()
        
        if 'content' in chunk:
            # Extract text from content
            content = chunk['content']
            text = extract_text_from_content(content)
            
            # Add the text field
            fixed_chunk['text'] = text
            
            # Keep content as source
            fixed_chunk['source_content'] = content
        else:
            # No content field
            fixed_chunk['text'] = ''
        
        # Ensure we have proper metadata
        if 'metadata' not in fixed_chunk:
            fixed_chunk['metadata'] = {}
        
        # Add source and ID to metadata
        fixed_chunk['metadata']['source'] = 'chunked_data'
        fixed_chunk['metadata']['chunk_id'] = chunk.get('chunk_id', '')
        
        fixed_chunks.append(fixed_chunk)
    
    # Save fixed chunks
    with open(output_file, 'w') as f:
        json.dump(fixed_chunks, f)
    
    print(f"Fixed chunks saved to {output_file}")
    print(f"Total chunks: {len(fixed_chunks)}")

if __name__ == "__main__":
    main() 