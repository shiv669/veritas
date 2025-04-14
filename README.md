# Veritas: LLM for Science

**Veritas** is an open-source, modular Large Language Model (LLM) built to revolutionize scientific research.  
Designed for transparency, adaptability, and epistemic trust, Veritas empowers researchers to access, verify, and synthesize scientific knowledge at scale — with full local control.

> Built on the Mistral-2 30B architecture, fine-tuned via QLoRA, and extended with 100K+ token context windows.

---

## Features

- **Citation-Backed Reasoning** – Generates answers with verifiable references using RAG.
- **100K+ Token Context** – Ingest entire research papers, books, or multi-document corpora using LongLoRA + positional interpolation.
- **Plug-and-Play Adapters** – Load domain-specific LoRA adapters (e.g., neuroscience, law) on demand.
- **Continual Learning** – Modular RAG pipeline allows weekly updates without retraining.
- **RLHF/DPO Alignment** – Fine-tuned with human feedback for scientific accuracy and citation quality.
