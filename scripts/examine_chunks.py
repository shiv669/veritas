#!/usr/bin/env python3
"""
Script to examine the structure of chunked_data.json
"""

import json
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
                
                # Go back to beginning and load a few items
                f.seek(0)
                # Try to load first 5 items
                chunks = []
                depth = 0
                brackets = 0
                chunk_data = ""
                
                # Read char by char looking for 5 complete JSON objects
                for char in f.read(100000):  # Read a larger chunk to find complete objects
                    chunk_data += char
                    
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and brackets > 0:
                            # Found a complete object
                            try:
                                chunks.append(json.loads(chunk_data.strip().rstrip(',').strip()))
                                chunk_data = ""
                                if len(chunks) >= 5:
                                    break
                            except:
                                chunk_data = ""
                    elif char == '[' and depth == 0:
                        brackets += 1
                        chunk_data = ""
                
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