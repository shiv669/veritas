# Veritas AI Scientist

Veritas AI Scientist is an advanced research assistant that helps generate research ideas, design experiments, and produce scientific writeups using the Mistral model with RAG capabilities.

## Features

- Generate novel research ideas based on templates
- Memory-optimized for M4 Mac (prevents OOM errors)
- Phased execution (run idea generation, experiments, or writeup separately)
- Wide range of research templates
- RAG-powered for up-to-date scientific knowledge

## Prerequisites

1. Veritas codebase with Mistral RAG system set up
2. Research templates in `models/Cognition`
3. Python 3.11+
4. Dependencies installed

## Installation

We provide a dependency installation script to set up all required packages:

```bash
# Navigate to the AI Scientist directory
cd src/veritas/ai_scientist

# Install core dependencies only
python install_dependencies.py

# Install both core and research template dependencies
python install_dependencies.py --all

# Upgrade existing packages
python install_dependencies.py --all --upgrade
```

## Quick Start

The easiest way to get started is to use the interactive interface:

```bash
# Navigate to the AI Scientist directory
cd src/veritas/ai_scientist

# Run the interactive interface
python run_interface.py
```

This will guide you through selecting a research template, number of ideas, and execution mode.

You can also run specific modes directly:

```bash
# Simple mode (fastest)
python run_interface.py --mode simple

# Optimized mode (recommended)
python run_interface.py --mode direct --experiment nanoGPT_lite --num-ideas 1

# Comprehensive mode (in-depth)
python run_interface.py --mode full --experiment nanoGPT_lite --num-ideas 1
```

## Alternative Methods

For more specific use cases, you can use these scripts directly:

```bash
# Simple test that generates one idea
python test_simple.py

# Run with optimized implementation
./run.sh --phase idea --num-ideas 1 --use-direct-implementation
```

## System Components

1. **LLM System**: Powers the AI Scientist with our Mistral model and RAG capabilities
2. **Memory Manager**: Handles memory optimization for M4 Mac
3. **Prompt Strategy**: Optimizes prompts for better results
4. **Research Templates**: Pre-defined research areas and examples
5. **Main Script**: Orchestrates the scientific workflow

## Troubleshooting

If you encounter issues with the Veritas AI Scientist:

1. Try running with `--use-direct-implementation` flag
2. Check the logs in the `logs` directory
3. Reduce number of ideas with `--num-ideas 1`
4. Check your PYTHONPATH includes all required directories
5. Run `python install_dependencies.py --all --upgrade` to ensure all dependencies are up to date

## Status

- The current system successfully handles idea generation
- Experiment execution and writeup phases are in development
- Memory optimization is fully functional on M4 Mac 