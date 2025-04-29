#!/usr/bin/env python3
"""
test_process.py

Test script to verify the entire chunk processing pipeline.
Tests all major components including chunk processing, metadata validation,
and RAG system integration.
"""

import json
import logging
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import pytest
from typing import List, Dict, Any
import time

from index_chunks_parallel import (
    process_chunk,
    validate_metadata,
    get_metadata_stats,
    backup_metadata,
    restore_metadata,
    process_chunk_group
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockRAGSystem:
    """Mock RAG system for testing."""
    def __init__(self):
        self.texts = []
        self.metadata = []
        
    def add_text(self, text: str, metadata: Dict[str, Any]):
        """Mock method to add text and metadata."""
        self.texts.append(text)
        self.metadata.append(metadata)
        return True

def create_test_chunks(num_chunks: int = 5) -> List[Dict[str, Any]]:
    """Create test chunks with various metadata configurations."""
    chunks = []
    for i in range(num_chunks):
        chunk = {
            'text': f'Test chunk {i} content',
            'metadata': {
                'source': f'test_source_{i}',
                'chunk_index': i,
                'timestamp': datetime.now().isoformat(),
                'document_id': f'doc_{i}',
                'title': f'Test Document {i}',
                'author': 'Test Author',
                'page_number': i + 1,
                'section': f'Section {i}',
                'language': 'en',
                'confidence_score': 0.95
            }
        }
        chunks.append(chunk)
    return chunks

def test_metadata_validation():
    """Test metadata validation function."""
    # Test with complete metadata
    complete_metadata = {
        'source': 'test_source',
        'chunk_index': 0,
        'timestamp': datetime.now().isoformat(),
        'document_id': 'doc_1',
        'title': 'Test Document',
        'author': 'Test Author',
        'page_number': 1,
        'section': 'Section 1',
        'language': 'en',
        'confidence_score': 0.95
    }
    validated = validate_metadata(complete_metadata, 0)
    assert validated == complete_metadata

    # Test with missing fields
    incomplete_metadata = {
        'source': 'test_source',
        'chunk_index': 1
    }
    validated = validate_metadata(incomplete_metadata, 1)
    assert validated['source'] == 'test_source'
    assert validated['chunk_index'] == 1
    assert 'timestamp' in validated
    assert 'document_id' in validated
    assert validated['language'] == 'en'
    assert validated['confidence_score'] == 1.0

    # Test with invalid types
    invalid_metadata = {
        'source': 123,
        'chunk_index': 'invalid',
        'page_number': 'not_a_number'
    }
    validated = validate_metadata(invalid_metadata, 2)
    assert isinstance(validated['source'], str)
    assert isinstance(validated['chunk_index'], int)
    assert isinstance(validated['page_number'], int)

def test_metadata_stats():
    """Test metadata statistics generation."""
    chunks = create_test_chunks(3)
    stats = get_metadata_stats(chunks)
    
    assert stats['total_chunks'] == 3
    assert 'field_counts' in stats
    assert 'field_values' in stats
    assert 'missing_fields' in stats
    assert 'validation_errors' in stats
    
    # Verify field counts
    assert stats['field_counts']['source'] == 3
    assert stats['field_counts']['language'] == 3
    
    # Verify field values
    assert len(stats['field_values']['source']) == 3
    assert 'en' in stats['field_values']['language']

def test_backup_restore():
    """Test metadata backup and restore functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_path = Path(temp_dir)
        chunks = create_test_chunks(3)
        
        # Extract metadata for backup
        metadata_list = [chunk['metadata'] for chunk in chunks]
        
        # Test backup
        success = backup_metadata(metadata_list, backup_path)
        assert success
        
        # Verify backup file exists
        backup_files = list(backup_path.glob("metadata_backup_*.json"))
        assert len(backup_files) == 1
        
        # Test restore
        restored = restore_metadata(backup_path)
        assert restored is not None
        assert len(restored) == len(chunks)
        
        # Verify restored data matches original
        for orig_meta, rest_meta in zip(metadata_list, restored):
            assert orig_meta['source'] == rest_meta['source']
            assert orig_meta['chunk_index'] == rest_meta['chunk_index']

def test_chunk_processing():
    """Test individual chunk processing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        rag = MockRAGSystem()
        
        # Test valid chunk
        chunk = create_test_chunks(1)[0]
        result = process_chunk(chunk, rag, output_dir)
        assert result is not None
        assert result['source'] == chunk['metadata']['source']
        assert len(rag.texts) == 1
        assert len(rag.metadata) == 1
        
        # Test invalid chunk (empty text)
        invalid_chunk = {
            'text': '',
            'metadata': {'source': 'test'}
        }
        result = process_chunk(invalid_chunk, rag, output_dir)
        assert result is None
        assert len(rag.texts) == 1  # Should not have added the invalid chunk

def test_group_processing():
    """Test processing of chunk groups."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        backup_dir = Path(temp_dir) / 'backups'
        rag = MockRAGSystem()
        
        # Create test chunks
        chunks = create_test_chunks(5)
        
        # Process group
        successful, failed = process_chunk_group(0, chunks, rag, output_dir, backup_dir)
        
        # Wait a bit for files to be written
        time.sleep(0.1)
        
        assert successful == len(chunks)
        assert failed == 0
        assert len(rag.texts) == len(chunks)
        assert len(rag.metadata) == len(chunks)
        
        # Debug: List all files in output directory
        logger.info(f"Output directory contents: {list(output_dir.glob('*'))}")
        
        # Verify output files
        output_files = list(output_dir.glob("chunk_*.json"))
        logger.info(f"Found output files: {output_files}")
        assert len(output_files) == len(chunks)
        
        # Verify backup was created
        backup_files = list(backup_dir.glob("metadata_backup_*.json"))
        assert len(backup_files) == 1

def main():
    """Run all tests."""
    logger.info("Starting process tests...")
    
    # Run tests
    test_metadata_validation()
    logger.info("✓ Metadata validation test passed")
    
    test_metadata_stats()
    logger.info("✓ Metadata statistics test passed")
    
    test_backup_restore()
    logger.info("✓ Backup and restore test passed")
    
    test_chunk_processing()
    logger.info("✓ Chunk processing test passed")
    
    test_group_processing()
    logger.info("✓ Group processing test passed")
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main() 