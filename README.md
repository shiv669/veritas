# Veritas: A Scientist for Autonomous Research

One of the grand challenges of artificial intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have been used to assist human scientists—for example, in brainstorming ideas or writing code—they still require extensive manual supervision or are constrained to narrow, task-specific use cases.

**Veritas** is a comprehensive system for fully automatic scientific discovery, enabling Foundation Models such as Large Language Models (LLMs) to perform research independently. It runs locally on **Mistral 7B**, ensuring full data privacy, minimizing citation hallucinations through **Retrieval-Augmented Generation (RAG)**, and supporting customizable scientific writing styles via **QLoRA (Quantized Low-Rank Adaptation)**. Veritas also integrates **LongLoRA** for context extension, allowing input windows of over **100,000 tokens** to support long-form research workflows.

Built during Major League Hacking's Open Source Hack Week, Veritas was selected among the top 1% of projects submitted globally.

## Example papers

Veritas has generated machine learning research papers in areas such as:

- DualScale Diffusion: Adaptive Feature Balancing for Low-Dimensional Generative Models- Language model architecture
- StyleFusion: Adaptive Multi-style Generation in Character-Level Language Models
- Adaptive Learning Rates for Transformers via Q-Learning
- Unlocking Grokking: A Comparative Study of Weight Initialization Strategies in Transformer Models

These papers were rated "Weak Accept" or higher by the system's built-in reviewer, based on major ML conference criteria.

## Core Research Workflow

Veritas mirrors the architecture of The AI Scientist and implements the full research pipeline:

![AI Scientist Architecture](https://github.com/user-attachments/assets/e166c52d-ac74-44b3-9e5a-78c4a3309492)

### 1. Idea Generation
- Receives a topic template
- Brainstorms novel research directions
- Validates novelty using Semantic Scholar

### 2. Experimental Iteration
- Executes code for proposed methods
- Collects outputs and visualizations
- Annotates each result for interpretation

### 3. Paper Write-up
- Generates a LaTeX-formatted scientific paper
- Autonomously sources relevant citations

### 4. Automated Peer Review
- Uses a custom LLM reviewer aligned with ML conference standards
- Evaluates novelty, clarity, rigor
- Feeds back into the system for future iterations

## Technical Stack

| Component           | Description                          |
|---------------------|--------------------------------------|
| Language Model      | Mistral 2 7B with QLoRA               |
| Context Extension   | LongLoRA (supports 100K+ tokens)      |
| Retrieval Engine    | FAISS + SentenceTransformers          |
| RAG Pipeline        | Custom Python + Hugging Face stack    |
| Hardware            | MacBook Pro M4 Max (local execution)  |

## Acknowledgements

- Mistral AI for the open-weight 7B model  
- Hugging Face for Transformers and SentenceTransformers  
- Facebook Research for FAISS  
- PyTorch for MPS support on Apple Silicon  
- Major League Hacking (MLH) for the hackathon platform  
- Sakana AI for pioneering The AI Scientist concept  
