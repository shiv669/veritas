# Veritas: A Scientist for Autonomous Research

**Veritas** is an open-source, modular Large Language Model (LLM) for scientific research.  

Designed for transparency, adaptability, and epistemic trust, Veritas empowers researchers to access, verify, and synthesize scientific knowledge at scale — with full local control.

> Built on the Mistral-2 7B architecture, fine-tuned via QLoRA, extended with 100K+ token context windows, and enhanced with a RAG (Retrieval-Augmented Generation) system for citation-backed reasoning.

---

## Features

- **Citation-Backed Reasoning** – Generates answers with verifiable references using RAG.
- **100K+ Token Context** – Ingest entire research papers, books, or multi-document corpora using LongLoRA + positional interpolation.
- **Plug-and-Play Adapters** – Load domain-specific LoRA adapters (e.g., neuroscience, law) on demand.
- **Continual Learning** – Modular RAG pipeline allows weekly updates without retraining.
- **RLHF/DPO Alignment** – Fine-tuned with human feedback for scientific accuracy and citation quality.

## Repository Structure

The Veritas project is organized as follows:

- `src/veritas/` - Core package implementation
  - `__init__.py` - Package initialization and exports
  - `config.py` - Configuration settings
  - `chunking.py` - Text chunking implementation
  - `rag.py` - RAG system implementation
  - `mps_utils.py` - MPS (Metal Performance Shaders) utilities
  - `utils.py` - General utility functions

- `scripts/` - Utility scripts
  - `chunk_data.py` - Text chunking script
  - `index_chunks.py` - Indexing script
  - `index_chunks_parallel.py` - Parallel indexing
  - `improved_chunking.py` - Enhanced chunking
  - `process_json.py` - JSON processing
  - `analyze_chunks.py` - Chunk analysis
  - `analyze_fulltext.py` - Full text analysis
  - `clean_encoding.py` - Encoding cleanup
  - `process_text.py` - Text processing
  - `cleanup.py` - Cleanup utilities

- `tests/` - Test suite
  - `test_rag_system.py` - RAG system tests
  - `test_rag_workflow.py` - Workflow tests
  - `test_rag_performance.py` - Performance tests
  - `test_rag_response.py` - Response quality tests
  - `test_mps_performance.py` - MPS tests
  - `test_process.py` - Processing tests
  - `test_rag.py` - RAG module tests
  - `test_chunking.py` - Chunking tests
  - `test_docs/` - Test documentation
  - `test_results/` - Test results

- `data/` - Data directory
  - `input/` - Input data
  - `output/` - Output data
  - `chunks/` - Processed chunks
  - `indices/` - FAISS indices

- `models/` - Model directory
  - `embeddings/` - Embedding models
  - `faiss/` - FAISS indices
  - `mistral/` - Mistral model files

- `docs/` - Documentation and resources
  - `chunk_size_distribution.png` - Chunk size visualization

- `logs/` - Log files

## Installation

To install the package in development mode:

```bash
pip install -e .
```

## Project Structure

```
veritas/
├── scripts/
│   ├── indexing/           # RAG indexing and document processing
│   │   ├── format_rag.py   # Document formatting and chunking
│   │   └── build_index.py  # FAISS index construction
│   └── training/           # Model training scripts
├── data/                   # Document storage and processed data
├── models/                 # Model checkpoints and adapters
└── logs/                   # Training and processing logs
```

## Usage

### Document Processing

1. Place your documents in the `data/` directory
2. Process documents for RAG:
```bash
python scripts/indexing/format_rag.py --input-dir data/ --output-file data/processed.json
```

3. Build the FAISS index:
```bash
python scripts/indexing/build_index.py --input-file data/processed.json
```

### Model Training

1. Prepare your training data
2. Run the training script:
```bash
python scripts/training/train.py --config configs/training_config.yaml
```

## Configuration

The project uses a modular configuration system. Key configuration files:

- `config.py` - Main configuration settings
- `configs/training_config.yaml` - Training parameters
- `configs/indexing_config.yaml` - RAG indexing settings
