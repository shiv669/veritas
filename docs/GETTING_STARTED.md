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

### Step 1: Set up your environment

Make sure you have Python 3.8 or higher installed. Then run:

```bash
# Clone the repository
git clone https://github.com/yourusername/veritas.git
cd veritas

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Step 2: Download the Mistral 2 7B model

```bash
# Create model directory
mkdir -p models/mistral

# Download model files (this may take a while)
python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-v0.2', local_dir='models/mistral')"
```

## Using Veritas

### Step 1: Add your documents

Place your research PDFs, text files, or other documents in the `data/input` directory.

### Step 2: Process your documents

```bash
# Create the directories if they don't exist
mkdir -p data/input data/output data/indices/latest

# Process your documents
python scripts/indexing/format_rag.py --input-dir data/input --output-file data/processed.json

# Build the search index
python scripts/indexing/build_faiss_index.py --input-file data/processed.json --output-dir data/indices/latest
```

### Step 3: Ask questions about your documents

```python
from veritas.rag import query_rag

# Ask a question
result = query_rag("What is the main conclusion of the paper?")
print(result["answer"])

# See where the information came from
for chunk in result["retrieved_chunks"]:
    print(f"Source: {chunk['chunk'].get('source', 'unknown')}")
```

## Using the AI Scientist

The AI Scientist component helps you generate research ideas, design experiments, and produce scientific content.

### Step 1: Navigate to the AI Scientist directory

```bash
cd src/veritas/ai_scientist
```

### Step 2: Run the interactive interface

```bash
python run_interface.py
```

This will guide you through:
- Selecting a research template
- Choosing the number of ideas to generate
- Setting the execution mode

### Step 3: Try different options

```bash
# Run with specific parameters
python run_interface.py --mode direct --experiment nanoGPT_lite --num-ideas 1

# Run a simple test
python test_simple.py

# Run all available tests
./test_all.sh
```

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

Need more help? Check the [README.md](../README.md) or open an issue on GitHub! 