# Getting Started with Veritas v1.2

Welcome to Veritas! This guide will help you get up and running with minimal technical knowledge.

## What is Veritas?

Veritas is an AI assistant for research that can:
- Answer questions based on your documents
- Provide citations to where it found information
- Process large collections of research papers, books, or technical documents
- Generate novel research ideas and scientific content

It's like having a research assistant who has read all your documents and can answer questions about them!

## Quick Installation

### Unified Installation (Recommended)

The fastest way to get started with Veritas is to use our unified installation script:

```bash
# Clone the repository
git clone https://github.com/yourusername/veritas.git
cd veritas

# Run the basic installation
python install.py

# To also download the Mistral model (optional, 13GB+)
python install.py --download-model
```

#### Advanced Installation Options

The installation script provides several options to customize your setup:

```bash
# Upgrade existing dependencies
python install.py --upgrade

# Continue installation even if some steps fail
python install.py --ignore-errors

# Skip dependency installation (useful for development)
python install.py --skip-dependencies

# Specify a different model to download
python install.py --download-model --model "mistralai/Mistral-7B-v0.2"

# See all available options
python install.py --help
```

The installation script will:
1. Create all necessary directories
2. Install all dependencies for both the RAG system and AI Scientist
3. Set up the Veritas package
4. Create basic research templates
5. Optionally download the Mistral model

After installation, you can use the command-line tools:
```bash
# Launch the RAG system (default)
veritas

# Launch the AI Scientist directly
veritas-ai-scientist
```

### Manual Installation

If you prefer to install manually:

#### Step 1: Set up your environment

Make sure you have Python 3.9 or higher installed. Then run:

```bash
# Clone the repository
git clone https://github.com/yourusername/veritas.git
cd veritas

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

#### Step 2: Download the Mistral 2 7B model

```bash
# Create model directory
mkdir -p models/mistral

# Download model files (this may take a while)
python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-v0.2', local_dir='models/mistral')"
```

## Using Veritas

### Unified Interface

Veritas now provides a unified interface to access both its RAG system and AI Scientist:

```bash
# Start the RAG system (default)
python scripts/run.py

# Start the AI Scientist directly
python scripts/run.py --system ai_scientist

# For help and more options
python scripts/run.py --help
```

### RAG System: Document Question Answering

#### Step 1: Add your documents

Place your research PDFs, text files, or other documents in the `data/input` directory.

#### Step 2: Process your documents

```bash
# Create the directories if they don't exist
mkdir -p data/input data/output data/indices/latest

# Process your documents
python scripts/indexing/format_rag.py --input-dir data/input --output-file data/processed.json

# Build the search index
python scripts/indexing/build_faiss_index.py --input-file data/processed.json --output-dir data/indices/latest
```

#### Step 3: Ask questions about your documents

Start the RAG system and ask questions through the terminal interface:

```bash
python scripts/run.py
```

Or programmatically:

```python
from veritas.rag import query_rag

# Ask a question
result = query_rag("What is the main conclusion of the paper?")
print(result["answer"])

# See where the information came from
for chunk in result["retrieved_chunks"]:
    print(f"Source: {chunk['chunk'].get('source', 'unknown')}")
```

### AI Scientist: Research Assistant

The AI Scientist component helps you generate research ideas, design experiments, and produce scientific content.

#### Method 1: Using the unified interface (recommended)

```bash
# Start the AI Scientist directly
python scripts/run.py --system ai_scientist
```

This will provide a menu-driven interface for accessing all AI Scientist features.

#### Method 2: Direct access to AI Scientist scripts

```bash
# Navigate to the AI Scientist directory
cd src/veritas/ai_scientist

# Run the interactive interface
python run_interface.py

# Run a simple test
python test_simple.py

# Run all available tests
./test_all.sh
```

#### Method 3: Switch between modes during a session

1. Start the RAG system: `python scripts/run.py`
2. Type `scientist` at the prompt to switch to AI Scientist mode
3. When finished with AI Scientist, select option 4 to return to RAG mode

For more advanced usage, see the [AI Scientist documentation](AI_SCIENTIST.md).

## Next Steps

- Try different question formats to get the best responses
- Add more documents to expand your knowledge base
- Experiment with different research templates
- Check out the more advanced features in the full documentation

## Troubleshooting

- **Out of memory errors**: Reduce batch size in the indexing script or use `--use-direct-implementation` flag
- **Model loading errors**: Make sure you downloaded the correct model
- **Empty responses**: Check that your documents were properly processed
- **Template errors**: Verify that the template path is correct in your configuration
- **Installation errors**: Make sure you're using Python 3.9 or higher and have sufficient permissions
- **Dependency errors**: Try running with `--ignore-errors` or check specific package compatibility

Need more help? Check the [README.md](../README.md) or open an issue on GitHub! 