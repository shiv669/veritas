# Veritas v1.0: A Scientist for Autonomous Research

**Veritas** is an open-source, modular Large Language Model (LLM) for scientific research.  

Designed for transparency, adaptability, and epistemic trust, Veritas empowers researchers to access, verify, and synthesize scientific knowledge at scale — with full local control.

## Architecture

Veritas is built on the **Mistral-2 7B** foundation model, enhanced with:
- Fine-tuning via QLoRA for domain adaptation
- 100K+ token context windows through LongLoRA + positional interpolation
- RAG (Retrieval-Augmented Generation) system for citation-backed reasoning

## Features

### Core Capabilities
- **Mistral 2 7B Architecture** – Built on one of the most powerful open-source language models
- **Citation-Backed Reasoning** – Generates answers with verifiable references using RAG
- **100K+ Token Context** – Process entire research papers, books, or multi-document corpora

### Research-Optimized Design
- **Scientific Knowledge Base** – Pre-indexed scientific literature for immediate research assistance
- **Specialized Retrieval** – Optimized for scientific papers, technical documents, and research materials
- **Domain-Specific Adapters** – Load specialized LoRA adapters for fields like neuroscience, law, or medicine

### Advanced Implementation
- **Modular RAG Pipeline** – Weekly knowledge updates without retraining
- **Optimized Performance** – Support for CUDA, MPS (Apple Silicon), and CPU deployment
- **Extensible Architecture** – Easy to customize for specific research domains

## Repository Structure

### Core Components

| Directory | Description |
|-----------|-------------|
| `src/veritas/` | Core implementation of Veritas system |
| `scripts/` | Utility scripts for processing, indexing and analysis |
| `data/` | Document storage and processed data |
| `models/` | Model files including Mistral 2 7B and embeddings |
| `tests/` | Comprehensive test suite |
| `docs/` | Documentation and resources |
| `logs/` | Log files from training and processing |

### Key Files

| File | Purpose |
|------|---------|
| `src/veritas/rag.py` | RAG system implementation |
| `src/veritas/config.py` | Configuration settings |
| `src/veritas/chunking.py` | Document chunking logic |
| `scripts/indexing/build_index.py` | FAISS index builder |
| `scripts/indexing/format_rag.py` | Document preprocessor |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 16GB RAM for inference (32GB+ recommended)
- For GPU acceleration: 
  - CUDA-compatible GPU with 8GB+ VRAM for Mistral 2 7B
  - Apple Silicon Mac for MPS acceleration

### Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/veritas.git
cd veritas
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Model Setup

Veritas requires the Mistral 2 7B model. You can set it up as follows:

```bash
# Create model directory
mkdir -p models/mistral

# Download Mistral 2 7B model files (adjust URL as needed)
python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-v0.2', local_dir='models/mistral')"
```

### Docker Installation (Alternative)

For containerized deployment:

```bash
docker build -t veritas:1.0 .
docker run -p 7860:7860 -v $(pwd)/data:/app/data veritas:1.0
```

## Project Structure

```
veritas/
├── src/
│   └── veritas/              # Core package implementation
│       ├── __init__.py       # Package initialization
│       ├── config.py         # Configuration settings
│       ├── chunking.py       # Text chunking implementation
│       ├── rag.py            # RAG system implementation
│       ├── mps_utils.py      # MPS (Metal Performance Shaders) utilities
│       └── utils.py          # General utility functions
├── scripts/
│   ├── indexing/             # RAG indexing and document processing
│   │   ├── format_rag.py     # Document formatting and chunking
│   │   └── build_index.py    # FAISS index construction
│   └── training/             # Model training scripts
├── data/                     # Document storage and processed data
│   ├── input/                # Input documents
│   ├── output/               # Processed output
│   ├── chunks/               # Text chunks for indexing
│   └── indices/              # FAISS indices
├── models/                   # Model files
│   ├── embeddings/           # Embedding models
│   ├── mistral/              # Mistral 2 7B model files
│   └── lora/                 # LoRA adapter files
└── logs/                     # Training and processing logs
```

## Usage

### Setting Up the Environment

1. Install the package in development mode:
```bash
pip install -e .
```

2. Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

### Document Processing for RAG

1. Place your research documents in the `data/input/` directory

2. Process documents for RAG:
```bash
python scripts/indexing/format_rag.py --input-dir data/input/ --output-file data/processed.json
```

3. Build the FAISS index:
```bash
python scripts/indexing/build_index.py --input-file data/processed.json --output-dir data/indices/latest
```

### Running the Veritas System

#### Query with RAG
```python
from veritas.rag import RAGSystem, query_rag

# Simple query with default settings
result = query_rag("What are the mechanisms of long-term potentiation?")
print(result["answer"])

# Advanced usage with custom parameters
rag = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="models/mistral",
    index_path="data/indices/latest"
)
result = rag.retrieve("What are the mechanisms of long-term potentiation?", top_k=7)
```

#### Using Long Context Windows
For processing papers or long documents:
```python
from veritas.rag import RAGSystem

rag = RAGSystem(llm_model="models/mistral")
with open("data/input/full_paper.txt", "r") as f:
    paper_text = f.read()
    
response = rag.generate(
    f"Summarize the following paper: {paper_text}",
    max_length=4096
)
print(response)
```

## Configuration

The project uses a modular configuration system. Key configuration files:

- `config.py` - Main configuration settings
- `configs/training_config.yaml` - Training parameters
- `configs/indexing_config.yaml` - RAG indexing settings

## Release Notes

### Version 1.0 (Current)
- Full RAG system implementation for citation-backed responses
- Support for Mistral 2 7B model architecture
- Enhanced retrieval capabilities for scientific literature
- Improved chunking and indexing for better information retrieval
