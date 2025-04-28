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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/veritas.git
cd veritas
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
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
