# Veritas: A Scientist for Autonomous Research

One of the grand challenges in artificial intelligence is building agents capable of conducting scientific research independently—generating new knowledge without constant human supervision. While frontier models like GPT or Claude have been used to assist researchers in tasks like brainstorming or code generation, they still rely heavily on manual prompting and domain-specific constraints.

Veritas represents a step forward: a modular, locally deployable system built to function as an autonomous research assistant.

## Architecting Autonomous Research

Veritas is a Retrieval-Augmented Generation (RAG) system optimized for Apple Silicon. It combines the power of Mistral 2 7B with a memory-efficient, high-throughput retrieval pipeline. Developed during MLH Global Hack Week: Open Source, Veritas was designed from the ground up to support scientific reasoning, literature comprehension, and hypothesis generation.

![image](https://github.com/user-attachments/assets/3f42a833-0c0a-4c2c-9dee-244fcb4ca56f)

At its core, Veritas includes:

### AI Scientist Mode
A high-level orchestration module that enables Veritas to operate as a self-directed scientific agent:

**Autonomous Ideation**: Generates novel research questions and hypotheses based on literature patterns

**Contextual Comprehension**: Maintains continuity across long academic texts using extended token context

**Literature Synthesis**: Summarizes, integrates, and cites key papers to support theoretical framing

**Experimental Design**: Suggests research methodologies, metrics, and potential datasets

**Self-Evaluation**: Scores proposals on dimensions such as novelty, feasibility, and impact

## Key Features

**Optimized for Apple Silicon**: Tailored for M1, M2, M3, and M4 chips using Metal Performance Shaders (MPS)

**Memory-Efficient Architecture**: Prevents out-of-memory errors while supporting long-context generation

**High-Quality RAG Implementation**: Precision-tuned document retrieval and context-aware reasoning

**Terminal-Based Interface**: Lightweight, transparent interaction without reliance on web frameworks

**Modular Architecture**: Separation between RAG engine, interface layer, and scientific reasoning core

**Unified Entry Point**: Supports both standard RAG queries and AI Scientist workflows in one interface

## Scientific Rigor and Transparency

Veritas is not a black box. Every research output is traceable, every citation is sourced, and every step is auditable. The system is designed to augment—rather than replace—human researchers, by automating repetitive cognitive tasks while preserving scientific rigor and critical oversight.

## System Requirements

- Apple Silicon Mac (M1, M2, M3, or M4)
- macOS Monterey or later
- Minimum 16GB RAM (32GB+ recommended, 128GB optimal for M4)
- At least 8GB of free SSD storage
- Python 3.9 or higher

## Installation

```bash
git clone https://github.com/yourusername/veritas.git
cd veritas
python install.py --download-model
```

The installation script handles dependencies, model setup, and package configuration automatically.



## Quick Start

```bash
# Launch RAG system
python scripts/run.py

# Launch AI Scientist
python scripts/run.py --system ai_scientist

# Switch between modes by typing 'scientist' or 'rag' at the prompt
```

## Architecture

**Core RAG Implementation** (`src/veritas/rag.py`) – Retrieval and generation engine

**Application Layer** (`scripts/run.py`) – Interface and configuration

**AI Scientist** (`src/veritas/ai_scientist`) – Autonomous research capabilities

**Apple Silicon Optimizations** (`src/veritas/mps_utils.py`) – Metal framework utilities



## Usage

```python
from veritas import RAGSystem

# Initialize system
rag = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="models/mistral-7b",
    device="mps"
)

# Generate response
response = rag.generate_rag_response(
    query="How does a RAG system work?",
    top_k=5,
    max_new_tokens=200
)
```



## Optimizations

**MPS Acceleration** – Metal Performance Shaders for faster computation

**Memory Management** – Prevents out-of-memory errors on Apple Silicon

**Half-Precision Computing** – FP16 optimization for better performance

**SSD Offloading** – Reduces RAM pressure through intelligent caching

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements


Mistral AI – for the Mistral 7B model, which powers the core reasoning capabilities of Veritas
Hugging Face – for maintaining the Transformers and SentenceTransformers libraries, essential to our pipeline
FAISS (Facebook Research) – for enabling efficient vector search and scalable retrieval
PyTorch – for supporting MPS acceleration on Apple Silicon, making local deployment feasible
Major League Hacking – for providing the space and community to initiate and develop this project
Sakana AI – for introducing The AI Scientist, which inspired and informed key functionalities in Veritas