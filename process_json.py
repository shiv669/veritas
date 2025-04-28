import json
import os
from pathlib import Path
from improved_chunking import ChunkingStrategy, ChunkingConfig, ImprovedChunker

def process_json_file(input_file, output_dir, chunks_file):
    """
    Process a file containing multiple JSON documents:
    1. Split into individual files
    2. Apply hierarchical chunking with improved parameters
    3. Save chunks to a separate file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize chunker with improved hierarchical strategy
    config = ChunkingConfig(
        strategy=ChunkingStrategy.HIERARCHICAL,
        chunk_size=512,  # Target chunk size
        overlap=50,      # Overlap between chunks
        min_chunk_size=150,  # Increased minimum chunk size to reduce small chunks
        max_chunk_size=1000, # Maximum chunk size
        respect_sections=True,  # Respect document sections
        max_chunks_per_doc=1000  # Limit maximum chunks per document to prevent over-chunking
    )
    chunker = ImprovedChunker(config)
    
    # Store all chunks
    all_chunks = []
    
    # Read the file line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                # Parse JSON document
                doc = json.loads(line)
                
                # Create a unique filename based on ID or title
                filename = f"doc_{line_num}.json"
                if doc.get('coreId'):
                    filename = f"doc_{doc['coreId']}.json"
                elif doc.get('title'):
                    # Create filename from title (first 50 chars, alphanumeric only)
                    safe_title = ''.join(c for c in doc['title'] if c.isalnum())[:50]
                    filename = f"doc_{safe_title}.json"
                
                # Write individual document to file
                output_file = os.path.join(output_dir, filename)
                with open(output_file, 'w', encoding='utf-8') as out:
                    json.dump(doc, out, indent=2, ensure_ascii=False)
                
                # Process document for chunking
                text = doc.get('fullText', '')
                if not text:
                    text = doc.get('abstract', '')
                
                if text:
                    # Create chunks
                    chunks = chunker.chunk_text(text)
                    
                    # Post-process chunks to merge very small ones
                    processed_chunks = merge_small_chunks(chunks, min_size=150)
                    
                    # Add metadata to chunks
                    for i, chunk in enumerate(processed_chunks):
                        chunk_data = {
                            'text': chunk,
                            'source': filename,
                            'chunk_index': i,
                            'total_chunks': len(processed_chunks),
                            'doc_id': doc.get('coreId', doc.get('doi', filename))
                        }
                        # Copy relevant metadata
                        for key in ['title', 'authors', 'abstract', 'year', 'doi', 'coreId']:
                            if key in doc:
                                chunk_data[key] = doc[key]
                        
                        all_chunks.append(chunk_data)
                
                line_num += 1
                if line_num % 100 == 0:
                    print(f"Processed {line_num} documents...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing document on line {line_num}: {e}")
                continue
    
    # Save all chunks to a single file
    print(f"Saving {len(all_chunks)} chunks to {chunks_file}...")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete! Processed {line_num} documents and created {len(all_chunks)} chunks.")

def merge_small_chunks(chunks, min_size=150):
    """
    Post-process chunks to merge very small ones with adjacent chunks.
    
    Args:
        chunks: List of text chunks
        min_size: Minimum size in words for a chunk
        
    Returns:
        List of processed chunks
    """
    if not chunks:
        return []
    
    result = []
    current_chunk = chunks[0]
    
    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        current_size = len(current_chunk.split())
        
        # If current chunk is small, merge with next chunk
        if current_size < min_size:
            current_chunk = current_chunk + " " + next_chunk
        else:
            # Current chunk is large enough, add it to result and move to next
            result.append(current_chunk)
            current_chunk = next_chunk
    
    # Add the last chunk
    result.append(current_chunk)
    
    return result

if __name__ == "__main__":
    input_file = "data/1.json"
    output_dir = "data/processed"
    chunks_file = "data/chunks.json"
    process_json_file(input_file, output_dir, chunks_file) 