#!/usr/bin/env python3
"""
Veritas CLI - Command-line interface for Veritas RAG system

This script provides a unified command-line interface for the Veritas RAG system,
consolidating functionality from multiple scripts into a single entry point.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.veritas.config import Config
from src.veritas.utils import setup_logging
from src.veritas.chunking import chunk_text
from src.veritas.rag import RAGSystem

# Configure logging
logger = setup_logging(__name__)

def process_command(args):
    """Process the 'process' command to preprocess and clean data."""
    if args.subcommand == 'json':
        from scripts.process_json import process_json
        process_json(args.input_file, args.output_file)
    elif args.subcommand == 'text':
        from scripts.process_text import process_text
        process_text(args.input_file, args.output_file, args.clean_encoding)
    elif args.subcommand == 'clean':
        from scripts.clean_encoding import clean_file
        clean_file(args.input_file, args.output_file)

def chunk_command(args):
    """Process the 'chunk' command to create text chunks."""
    from scripts.chunk_data import process_json_file
    process_json_file(args.input_file, args.output_dir, args.chunk_size, args.overlap)

def index_command(args):
    """Process the 'index' command to index chunks."""
    if args.parallel:
        from scripts.index_chunks_parallel import main as index_parallel
        index_parallel()
    else:
        from scripts.index_chunks import main as index_basic
        index_basic()

def analyze_command(args):
    """Process the 'analyze' command to analyze chunks or text."""
    if args.type == 'chunks':
        from scripts.analyze_chunks import main as analyze_chunks
        analyze_chunks()
    elif args.type == 'fulltext':
        from scripts.analyze_fulltext import main as analyze_fulltext
        analyze_fulltext()

def rag_command(args):
    """Process the 'rag' command to run the RAG system."""
    if args.mode == 'build':
        from scripts.build_rag import main as build_rag
        build_rag()
    elif args.mode == 'run':
        from scripts.run import main as run_rag
        run_rag()
    elif args.mode == 'query':
        # Direct query mode
        system = RAGSystem()
        index_path = os.path.join(Config.INDICES_DIR, "latest")
        system.load_index(index_path)
        results = system.retrieve(args.query, k=args.top_k)
        
        print(f"\nQuery: {args.query}")
        print("\nRetrieved chunks:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"])

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Veritas - RAG system command-line interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Preprocess and clean data")
    process_subparsers = process_parser.add_subparsers(dest="subcommand", help="Processing type")
    
    json_parser = process_subparsers.add_parser("json", help="Process JSON data")
    json_parser.add_argument("--input-file", "-i", required=True, help="Input JSON file")
    json_parser.add_argument("--output-file", "-o", required=True, help="Output JSON file")
    
    text_parser = process_subparsers.add_parser("text", help="Process text data")
    text_parser.add_argument("--input-file", "-i", required=True, help="Input text file")
    text_parser.add_argument("--output-file", "-o", required=True, help="Output text file")
    text_parser.add_argument("--clean-encoding", "-c", action="store_true", help="Clean encoding issues")
    
    clean_parser = process_subparsers.add_parser("clean", help="Clean encoding issues")
    clean_parser.add_argument("--input-file", "-i", required=True, help="Input file to clean")
    clean_parser.add_argument("--output-file", "-o", required=True, help="Output cleaned file")
    
    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Create text chunks")
    chunk_parser.add_argument("--input-file", "-i", required=True, help="Input file to chunk")
    chunk_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    chunk_parser.add_argument("--chunk-size", "-s", type=int, default=Config.DEFAULT_CHUNK_SIZE, help="Chunk size")
    chunk_parser.add_argument("--overlap", "-v", type=int, default=Config.DEFAULT_CHUNK_OVERLAP, help="Chunk overlap")
    chunk_parser.add_argument("--strategy", "-t", choices=["basic", "improved"], default="improved", help="Chunking strategy")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index chunks")
    index_parser.add_argument("--parallel", "-p", action="store_true", help="Use parallel indexing")
    index_parser.add_argument("--input-file", "-i", help="Input chunks file")
    index_parser.add_argument("--output-dir", "-o", help="Output directory")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze text or chunks")
    analyze_parser.add_argument("--type", "-t", choices=["chunks", "fulltext"], required=True, help="Analysis type")
    
    # RAG command
    rag_parser = subparsers.add_parser("rag", help="Run RAG system")
    rag_parser.add_argument("--mode", "-m", choices=["build", "run", "query"], required=True, help="RAG mode")
    rag_parser.add_argument("--query", "-q", help="Query for RAG system")
    rag_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of top results")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "process":
        process_command(args)
    elif args.command == "chunk":
        chunk_command(args)
    elif args.command == "index":
        index_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "rag":
        rag_command(args)
    else:
        print("Please specify a command. Use --help for more information.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 