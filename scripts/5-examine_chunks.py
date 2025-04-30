#!/usr/bin/env python3
"""
Script to examine the structure of chunked_data.json
"""

import json
import ijson
import os
from pathlib import Path
import sys

def main():
    """Main function to examine chunks"""
    # Path to chunks file
    chunks_file = Path("data/chunks/chunked_data.json")
    
    if not chunks_file.exists():
        print(f"Chunks file not found: {chunks_file}")
        return
    
    print(f"Examining {chunks_file}...")
    
    # Load the first part of the file to avoid loading everything
    with open(chunks_file, 'r') as f:
        # Read first 10000 characters
        data = f.read(10000)
        
        # Try to parse the beginning of the file
        try:
            # Check if it starts with a list
            if data.strip().startswith('['):
                print("File appears to be a JSON array")
                
                # Use ijson to stream-load first 5 chunks
                chunks = []
                with open(chunks_file, 'rb') as arr_f:
                    for obj in ijson.items(arr_f, 'item'):
                        chunks.append(obj)
                        if len(chunks) >= 5:
                            break

            # Check if it starts with an object
            elif data.strip().startswith('{'):
                print("File appears to be a JSON object")
                
                # Go back to beginning and load the structure
                f.seek(0)
                # Try to parse the structure (keys only)
                structure = json.loads(f.read(50000))
                if isinstance(structure, dict):
                    print(f"Top-level keys: {list(structure.keys())}")
                    
                    # If it has a chunks or documents key, examine one
                    for key in ['chunks', 'documents', 'data', 'items']:
                        if key in structure:
                            if isinstance(structure[key], list) and len(structure[key]) > 0:
                                chunks = structure[key][:5]
                                break
                    else:
                        # Take some values as samples
                        chunks = list(structure.values())[:5]
            else:
                print(f"Unknown format: starts with {data[:100]}")
                return
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"First 200 characters: {data[:200]}")
            return
    
    # Print sample chunk information
    print(f"\nFound {len(chunks)} sample chunks to examine")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        if isinstance(chunk, dict):
            print(f"  Keys: {list(chunk.keys())}")
            
            # Check for text content in different fields
            text_found = False
            
            # Try standard text fields
            for field in ['text', 'content', 'body', 'document', 'passage']:
                if field in chunk and chunk[field]:
                    print(f"  Content found in '{field}' field:")
                    print(f"    {chunk[field][:100]}...")
                    text_found = True
                    break
            
            # Check for metadata
            if 'metadata' in chunk:
                print(f"  Metadata keys: {list(chunk['metadata'].keys())}")
                
                # Check for text in metadata
                for field in ['text', 'content', 'body', 'document', 'passage']:
                    if field in chunk['metadata'] and chunk['metadata'][field]:
                        print(f"  Content found in 'metadata.{field}' field:")
                        print(f"    {chunk['metadata'][field][:100]}...")
                        text_found = True
                        break
            
            if not text_found:
                print("  No text content found in standard fields")
        else:
            print(f"  Not a dictionary: {type(chunk)}")
            print(f"  Value: {chunk}")

if __name__ == "__main__":
    main() 