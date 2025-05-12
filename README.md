# Veritas: High-Performance RAG for Apple Silicon

Veritas is a Retrieval-Augmented Generation (RAG) system optimized specifically for Apple Silicon M-series chips, with particular focus on the M4 Mac with 128GB RAM. It provides efficient document retrieval and context-aware answer generation using Mistral 2 7B.

## 🚀 Key Features

- **Apple Silicon Optimized**: Specially tuned for M1, M2, M3, and M4 Macs with MPS (Metal Performance Shaders) acceleration
- **Memory-Efficient Design**: Carefully manages memory to prevent OOM errors even with large models
- **High-Quality RAG**: Accurate document retrieval and context-aware answer generation
- **Terminal Interface**: Clean, simple interface for direct interaction without web frameworks
- **Modular Architecture**: Clear separation between core RAG implementation and application layer
- **AI Scientist**: Advanced research assistant built on top of our Mistral model with RAG capabilities
- **Unified Interface**: Access both RAG and AI Scientist from a single entry point

## 🔧 System Requirements

- Apple Silicon Mac (M1, M2, M3, or M4)
- macOS Monterey or later
- 16GB RAM minimum (32GB+ recommended, 128GB optimal for M4)
- 8GB+ free storage (SSD recommended)
- Python 3.9 or higher

## 📦 Installation

We provide a unified installation script that handles everything for you:

```bash
# Clone the repository
git clone https://github.com/yourusername/veritas.git
cd veritas

# Basic installation
python install.py

# To also download the Mistral model (optional, 13GB+)
python install.py --download-model

# More installation options
python install.py --upgrade                 # Upgrade existing dependencies
python install.py --ignore-errors           # Continue even if some steps fail
python install.py --skip-dependencies       # Skip installing dependencies
python install.py --model "mistralai/Mistral-7B-v0.2"  # Specify model to download
```

The installation script:
1. Creates necessary directories
2. Installs all dependencies for both RAG and AI Scientist
3. Sets up the package for development
4. Creates basic research templates for AI Scientist
5. Optionally downloads the Mistral model

After installation, you can use the command-line tools:
```bash
# Use main interface
veritas

# Use AI Scientist directly
veritas-ai-scientist

# See all available options
veritas --help
```

### Manual Installation

If you prefer manual installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/veritas.git
   cd veritas
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

4. Download and prepare the Mistral model (if needed):
   ```bash
   mkdir -p models/mistral
   python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-v0.2', local_dir='models/mistral')"
   ```

## 🔍 Quick Start

Run the unified terminal interface:

```bash
# Start with RAG system (default)
python scripts/run.py

# Start with AI Scientist
python scripts/run.py --system ai_scientist

# Show all options
python scripts/run.py --help
```

### Using the RAG System

The RAG system allows you to ask questions about your documents:

```bash
python scripts/run.py
```

This will start the RAG system with the terminal UI, where you can directly ask questions.

### Using AI Scientist

To use the AI Scientist component:

```bash
# Direct launch
python scripts/run.py --system ai_scientist

# Or start with RAG and switch
python scripts/run.py
# Then type 'scientist' at the prompt
```

Or run a simple test:

```bash
# Navigate to the AI Scientist directory
cd src/veritas/ai_scientist

# Simple test that generates one idea
python test_simple.py
```

For more information, see the [AI Scientist README](src/veritas/ai_scientist/README.md).

## 🏗️ Architecture

Veritas is designed with a clear separation of concerns:

- **Core RAG Implementation** (`src/veritas/rag.py`): The heart of the system that handles retrieval and generation
- **Application Layer** (`scripts/run.py`): Configures and uses the core RAG system for specific use cases
- **Configuration** (`src/veritas/config.py`): Centralized settings for the entire system
- **Apple Silicon Optimizations** (`src/veritas/mps_utils.py`): Specialized utilities for Apple's Metal framework
- **Text Processing** (`src/veritas/chunking.py`): Document segmentation for efficient indexing and retrieval
- **AI Scientist** (`src/veritas/ai_scientist`): Research assistant built on top of our RAG system

### UML Class Diagram

```
┌─────────────┐     ┌───────────────┐
│ MistralModel│     │   RAGSystem   │
│ (run.py)    │────>│  (rag.py)     │
└─────────────┘     └───────────────┘
       │                   │
       │                   │
       ▼                   ▼
┌─────────────┐     ┌───────────────┐
│ ModelConfig │     │    Config     │
└─────────────┘     └───────────────┘
                           │
                           ▼
                    ┌───────────────┐
                    │  mps_utils    │
                    └───────────────┘
```

## 🧩 Core Components

### RAGSystem (src/veritas/rag.py)

The main class that implements the RAG functionality:

```python
from veritas import RAGSystem

# Create a RAG system
rag = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="models/mistral-7b",
    index_path="models/faiss",
    device="mps"  # Use Apple Silicon acceleration
)

# Generate a complete RAG response
response = rag.generate_rag_response(
    query="How does a RAG system work?",
    top_k=5,  # Number of chunks to retrieve
    max_new_tokens=200
)

print(response["combined_response"])
```

### MistralModel (scripts/run.py)

A wrapper around RAGSystem that handles configuration and initialization:

```python
from src.veritas.config import Config
from scripts.run import MistralModel, ModelConfig

# Configure the model
config = ModelConfig(
    model_name=Config.LLM_MODEL,
    max_new_tokens=200,
    temperature=0.3,
    max_retrieved_chunks=3
)

# Create and load model
model = MistralModel(config)
model.load()

# Generate a response with context
context, direct_response, combined_response = model.generate(
    "What are the advantages of RAG systems over pure LLMs?"
)
```

### AI Scientist (src/veritas/ai_scientist)

A research assistant built on top of our Mistral model with RAG capabilities:

```python
from src.veritas.ai_scientist.run_scientist import AIScientist

# Create an AI Scientist instance
scientist = AIScientist(
    experiment="nanoGPT_lite", 
    num_ideas=1
)

# Generate research ideas
ideas = scientist.generate_ideas()

# Print the generated ideas
for idea in ideas:
    print(f"Idea: {idea['title']}")
    print(f"Description: {idea['description']}")
    print(f"Novelty: {idea['novelty_score']}")
```

## 🚴‍♀️ Advanced Usage

### Custom Document Chunking

```python
from veritas import chunk_text, get_chunk_size

# Get optimal chunk size based on document length
document_length = len(large_document)
chunk_size = get_chunk_size(document_length, target_chunks=20)

# Generate chunks with custom parameters
chunks = chunk_text(
    text=large_document,
    chunk_size=chunk_size,
    overlap=100  # Words of overlap between chunks
)

# Process each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:50]}...")
```

### Memory Optimization

```python
from veritas.mps_utils import optimize_memory_for_m4, clear_mps_cache

# Apply comprehensive M4 optimizations at startup
optimize_memory_for_m4()

# Clear cache after heavy operations
result = model.generate(complex_query)
clear_mps_cache()  # Free up GPU memory
```

### Switching Between RAG and AI Scientist

The unified interface allows you to switch between modes during a session:

```bash
# Start with RAG
python scripts/run.py

# Type 'scientist' at the prompt to switch to AI Scientist mode
# Select option 4 to return to RAG mode
```

## 📊 Performance Optimization

Veritas includes several optimizations for Apple Silicon:

1. **MPS Acceleration**: Uses Metal Performance Shaders for faster computation
2. **Memory Management**: Carefully controls memory usage to prevent OOM errors
3. **Half-Precision**: Uses FP16 where possible for better performance
4. **Caching Control**: Explicit cache clearing to prevent memory leaks
5. **SSD Offloading**: Uses SSD for temporary files to reduce RAM pressure

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Mistral AI for their excellent 7B model
- Hugging Face for Transformers and SentenceTransformers
- Facebook Research for FAISS
- The PyTorch team for MPS support
