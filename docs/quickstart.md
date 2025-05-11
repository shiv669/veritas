# Veritas Quickstart Guide

This guide provides practical, step-by-step instructions for getting started with Veritas.

## Initial Setup

Before using Veritas, ensure you have:

1. Installed all dependencies: `pip install -r requirements.txt`
2. Downloaded the Mistral 7B model to `models/mistral-7b/`
3. Created necessary data directories:
   ```bash
   mkdir -p data/input data/processed data/chunks models/faiss
   ```

## End-to-End Example

Let's walk through a complete example from adding documents to querying the system.

### 1. Prepare Your Documents

Place your documents in the `data/input` directory. Veritas supports several formats:

- **JSON files**: Collections of documents with metadata
- **Text files**: Plain text documents
- **PDF files**: Through the pdf processing script (requires pdf2text)

For this example, we'll create a simple text file:

```bash
# Create a sample document
echo "Unions and Workplace Safety: A Study

Research has shown that workplaces with union representation have 15-20% lower rates of serious injuries compared to non-unionized workplaces in the same industry. This is attributed to several factors:

1. Collective bargaining agreements often include safety provisions
2. Union workers receive more safety training
3. Union representatives can raise safety concerns without fear of retaliation
4. Unionized workplaces have higher compliance with OSHA regulations

However, the study also notes that this effect varies by industry, with the strongest effects seen in manufacturing and construction." > data/input/safety_study.txt
```

### 2. Process Your Documents

Next, we'll process the document to prepare it for chunking:

```bash
python scripts/cli.py process text --input-file data/input/safety_study.txt --output-file data/processed/safety_study.txt
```

### 3. Create Text Chunks

Now we'll split the processed document into chunks for indexing:

```bash
python scripts/cli.py chunk --input-file data/processed/safety_study.txt --output-dir data/chunks --chunk-size 500 --overlap 50
```

### 4. Build the Index

Create the FAISS index from the chunks:

```bash
python scripts/cli.py index --input-file data/chunks/chunks.json --output-dir models/faiss
```

### 5. Run the RAG System

Now you can start the interactive terminal interface:

```bash
python scripts/cli.py rag --mode run
```

Or make a direct query:

```bash
python scripts/cli.py rag --mode query --query "What effect do unions have on workplace safety?" --top-k 3
```

## Common Scenarios

### Adding New Documents

To add new documents to your existing index:

1. Add documents to `data/input/`
2. Process them as shown above
3. Rebuild the index with the `--append` flag:
   ```bash
   python scripts/cli.py index --input-file data/chunks/new_chunks.json --output-dir models/faiss --append
   ```

### Memory Optimization

If you're experiencing memory issues:

1. Reduce chunk size when building index
2. Lower the number of retrieved chunks
3. Edit `scripts/run.py` to adjust:
   ```python
   # In ModelConfig class
   max_new_tokens: int = 150  # Lower value
   max_retrieved_chunks: int = 1  # Retrieve fewer chunks
   ```

### Debugging

If your queries aren't returning expected results:

1. Examine the chunks directly:
   ```bash
   python scripts/cli.py analyze --type chunks
   ```

2. Use verbose querying:
   ```bash
   python scripts/cli.py rag --mode query --query "your query" --verbose
   ```

3. Check logs for detailed information:
   ```bash
   cat logs/mistral.log
   ```

## Next Steps

- Experiment with different chunking strategies for your specific documents
- Fine-tune the model configuration parameters
- Explore the API options for integrating Veritas into your applications

For more detailed information, refer to the project README and source code documentation. 