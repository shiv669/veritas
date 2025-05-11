# Veritas: Research Assistant with RAG

Veritas is a modular Retrieval-Augmented Generation (RAG) system powered by Mistral 2 7B, designed to help researchers access, verify, and synthesize scientific knowledge with full local control.

![Veritas Logo](docs/images/logo.png)

## Overview

Veritas provides citation-backed responses by retrieving relevant information from your document collection and using Mistral 2 7B to generate accurate answers with proper context.

**Key Features:**
- Local Mistral 2 7B inference with MPS support for Apple Silicon
- RAG system optimized for scientific research documents
- Terminal UI for easy interaction
- Memory-optimized for M4 Mac with up to 128GB RAM

## Quick Start

### Prerequisites

- Python 3.8+
- 16GB+ RAM (32GB+ recommended)
- For Apple Silicon: macOS with M1/M2/M3/M4 chip
- CUDA-compatible GPU for non-Mac systems

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/veritas.git
   cd veritas
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Mistral 2 7B model:**
   ```bash
   mkdir -p models/mistral-7b
   python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-v0.2', local_dir='models/mistral-7b')"
   ```

## Usage Guide

### Document Processing Pipeline

Veritas works by processing your documents through several stages:

1. **Processing**: Clean and prepare your documents
2. **Chunking**: Split documents into manageable chunks
3. **Indexing**: Create a searchable FAISS index
4. **Query**: Ask questions against your document collection

### Step 1: Add Your Documents

Place your research documents in the `data/input/` directory.

### Step 2: Process Your Documents

The CLI provides a unified interface for all operations:

```bash
# Process JSON documents
python scripts/cli.py process json --input-file data/input/documents.json --output-file data/processed/cleaned.json

# Or process text documents
python scripts/cli.py process text --input-file data/input/paper.txt --output-file data/processed/cleaned.txt
```

### Step 3: Create Text Chunks

```bash
python scripts/cli.py chunk --input-file data/processed/cleaned.json --output-dir data/chunks --chunk-size 1000 --overlap 100
```

### Step 4: Build the FAISS Index

```bash
python scripts/cli.py index --parallel
```

### Step 5: Run the RAG System

```bash
python scripts/cli.py rag --mode run
```

This launches the interactive terminal interface where you can query your documents.

### Direct Query Mode

For scripted queries, use the direct query mode:

```bash
python scripts/cli.py rag --mode query --query "What was the main finding of the study about unions and workplace safety?" --top-k 3
```

## Advanced Configuration

### Memory Optimization

For machines with different RAM configurations, edit the environment variables in `scripts/run.py`:

```python
os.environ.update({
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable upper limit to prevent OOM
    'PYTORCH_MPS_MEMORY_LIMIT': '80GB',  # Adjust based on your RAM
})
```

### Model Configuration

Adjust model parameters in `scripts/run.py` under the `ModelConfig` class:

```python
@dataclass
class ModelConfig:
    max_new_tokens: int = 200     # Reduce for memory savings
    temperature: float = 0.3      # Lower for more deterministic answers
    max_retrieved_chunks: int = 2 # Increase for more context (more memory)
```

## Troubleshooting

### Common Issues

- **Memory Errors**: Reduce `max_new_tokens` and `max_retrieved_chunks` in ModelConfig
- **Index Not Found**: Ensure your indexing process completed successfully
- **Model Loading Errors**: Verify Mistral 2 7B is downloaded correctly
- **High CPU Usage**: Lower process priority with `p.nice(15)` in run.py

## Directory Structure

```
veritas/
├── data/
│   ├── input/          # Your source documents
│   ├── processed/      # Cleaned documents
│   └── chunks/         # Text chunks for indexing
├── models/
│   ├── faiss/          # FAISS index and chunks
│   └── mistral-7b/     # Mistral 2 7B model files
├── scripts/
│   ├── cli.py          # Command line interface
│   └── run.py          # Core execution script
└── src/veritas/
    ├── config.py       # Configuration settings
    ├── rag.py          # RAG implementation
    └── chunking.py     # Document chunking logic
```

## Changelog

### v1.1 (Current)
- Optimized for M4 Max with 128GB RAM
- Memory-efficient RAG implementation
- Improved context chunking for better responses
- Terminal UI with dual-generation approach
- Fixed OOM errors with MPS backend

### v1.0
- Initial implementation with basic RAG capabilities
- Support for Mistral 2 7B model
- Document processing pipeline
- FAISS indexing for efficient retrieval

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mistral AI for the Mistral 2 7B model
- FAISS for vector search capabilities
- Sentence Transformers for document embeddings
