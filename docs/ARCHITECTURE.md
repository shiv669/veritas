# Veritas Architecture: How It All Works

This document explains how Veritas works in simple terms, without diving too deep into technical details.

## The Big Picture

Veritas combines two powerful technologies:

1. **Mistral 2 7B** - A large language model (LLM) that can generate human-like text
2. **RAG (Retrieval-Augmented Generation)** - A system that helps the model find and use information from your documents

Together, they create an AI that can answer questions based on your specific documents.

## How It Works: The Simple Version

Here's how Veritas answers your questions:

1. **Indexing Phase** (happens once, before you ask questions)
   - Your documents are broken into small chunks
   - Each chunk is converted into a special format (vectors) that the AI can search quickly
   - These chunks and vectors are organized into a searchable index

2. **Question Answering Phase** (happens every time you ask a question)
   - Your question is converted to the same special format (vector)
   - The system finds chunks that are most relevant to your question
   - These relevant chunks are sent to the Mistral 2 7B model
   - The model reads the chunks and generates an answer

## Key Components Explained

### 1. Text Chunking
Documents are too long to process all at once, so we break them into smaller pieces. Think of it like dividing a book into chapters or paragraphs that can be individually searched.

### 2. Embeddings
We convert text into numbers (vectors) that represent meaning. This allows the computer to understand that "automobile" and "car" are related concepts, even though they're different words.

### 3. FAISS Index
A special database that can quickly find similar chunks based on meaning, not just keywords. It's what makes searching through thousands of documents nearly instant.

### 4. Mistral 2 7B Model
The powerful AI that generates answers. It reads the context we provide (from your documents) and creates a human-like response.

### 5. RAG System
The orchestrator that connects all these components together into a seamless question-answering system.

## Technical Flow Diagram

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Your         │     │  Chunking     │     │  Embedding    │
│  Documents    │────▶│  System       │────▶│  System       │
└───────────────┘     └───────────────┘     └───────────────┘
                                                    │
                                                    ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Generated    │     │  Mistral 2    │     │  FAISS        │
│  Answer       │◀────│  7B Model     │◀────│  Index        │
└───────────────┘     └───────────────┘     └───────────────┘
        ▲                     ▲                     ▲
        │                     │                     │
        └─────────────┬───────┘                     │
                      │                             │
              ┌───────────────┐            ┌────────────────┐
              │  Your         │            │  Retrieval     │
              │  Question     │───────────▶│  System        │
              └───────────────┘            └────────────────┘
```

## Optimizations

Veritas includes special optimizations for:

- **Apple Silicon** - Special code to run faster on M-series Macs
- **Memory Management** - Efficient processing of large document collections
- **Parallel Processing** - Using multiple CPU cores for faster indexing

## Extensibility

The system is designed to be modular so you can:

- Add different document types (PDFs, web pages, etc.)
- Replace components with newer technologies as they become available
- Fine-tune the model for specific domains (medicine, law, etc.) 