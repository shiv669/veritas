# Veritas RAG System - Scripts

This directory contains utility scripts for the Veritas RAG system.

## Command-Line Interface

The main entry point is the `cli.py` script, which provides a unified command-line interface for all functionality.

### Installation

To install the CLI as a command-line tool:

```bash
cd scripts
pip install -e .
```

This will make the `veritas` command available in your environment.

### Usage

```
veritas <command> [options]
```

Available commands:

- `process` - Preprocess and clean data
  - `json` - Process JSON data
  - `text` - Process text data
  - `clean` - Clean encoding issues
- `chunk` - Create text chunks
- `index` - Index chunks
- `analyze` - Analyze text or chunks
- `rag` - Run RAG system
  - `build` - Build RAG system
  - `run` - Run RAG server
  - `query` - Query RAG system

For help on a specific command:

```bash
veritas <command> --help
```

## Directory Structure

- `scripts/` - Main scripts directory
  - `cli.py` - Main command-line interface
  - `setup.py` - Setup script for CLI installation
  - `indexing/` - Indexing-specific scripts
  - `retrieval/` - Retrieval-specific scripts
  - `archive/` - Archived/deprecated scripts

## Examples

### Process and chunk data

```bash
# Process a JSON file
veritas process json -i data/input/document.json -o data/output/processed.json

# Create chunks from a processed file
veritas chunk -i data/output/processed.json -o data/chunks/ -s 512 -v 128
```

### Build and query the RAG system

```bash
# Build the RAG index
veritas rag build

# Query the RAG system
veritas rag query -q "What is the main topic of the document?" -k 5
```

### Analyze chunk distribution

```bash
# Analyze chunks
veritas analyze -t chunks
``` 