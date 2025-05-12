# AI Scientist: Research Assistant Module

The AI Scientist is a core component of Veritas that provides automated research assistance capabilities. It helps generate research ideas, design experiments, and produce scientific writeups using our Mistral model with RAG capabilities.

## Features

- **Idea Generation**: Create novel research ideas based on templates
- **Memory Optimization**: Specially optimized for M4 Mac to prevent OOM errors
- **Phased Execution**: Run idea generation, experiments, or writeup phases separately
- **Template-Based**: Wide range of research templates for different domains
- **RAG-Powered**: Access to up-to-date scientific knowledge

## Architecture

The AI Scientist consists of several key components:

1. **Mistral Adapter**: Core interface to our Mistral model
2. **Memory Manager**: Handles efficient memory usage on Apple Silicon
3. **Prompt Strategy**: Creates optimized prompts for research tasks
4. **Research Templates**: Pre-defined domain-specific templates
5. **Main System**: Orchestrates the research workflow

## Usage Examples

### Basic Usage

```python
from veritas.ai_scientist.run_interface import run_interactive

# Run the interactive interface
run_interactive()
```

### Programmatic Usage

```python
from veritas.ai_scientist.run_scientist import AIScientist

# Create an AI Scientist instance
scientist = AIScientist(
    experiment="nanoGPT_lite",  # Template name
    num_ideas=3,                # Number of ideas to generate
    temperature=0.7,            # Creativity level
    mode="direct"               # Execution mode
)

# Generate research ideas
ideas = scientist.generate_ideas()

# Print the generated ideas
for i, idea in enumerate(ideas):
    print(f"Idea {i+1}: {idea['title']}")
    print(f"Description: {idea['description']}")
    print(f"Novelty: {idea['novelty_score']}")
```

### Running Specific Phases

```python
from veritas.ai_scientist.run_scientist import AIScientist

# Create an AI Scientist instance
scientist = AIScientist(experiment="nanoGPT_lite")

# Phase 1: Generate ideas
ideas = scientist.generate_ideas()

# Save the best idea for further exploration
best_idea = ideas[0]

# Phase 2: Design an experiment (when implemented)
experiment = scientist.design_experiment(idea=best_idea)

# Phase 3: Generate a writeup (when implemented)
writeup = scientist.generate_writeup(
    idea=best_idea,
    experiment=experiment
)
```

## Memory Management

The AI Scientist includes special memory optimization for Apple Silicon:

```python
from veritas.ai_scientist.memory_manager import MemoryManager

# Create a memory manager
memory_manager = MemoryManager()

# Apply optimizations before heavy operations
memory_manager.optimize_for_generation()

# Clear memory after heavy operations
memory_manager.clear_all_caches()
```

## Testing

The AI Scientist includes comprehensive testing capabilities:

```bash
# Run all tests
cd src/veritas/ai_scientist
./test_all.sh

# Run a specific test
python test_system.py --test-type=basic
```

## Troubleshooting

If you encounter issues with the AI Scientist:

1. **Memory Issues**: Try running with the `--use-direct-implementation` flag
2. **Slow Performance**: Reduce the number of ideas with `--num-ideas 1`
3. **Template Errors**: Check that the template path is correct
4. **Import Errors**: Ensure your PYTHONPATH includes the src directory

## Implementation Details

The AI Scientist uses a carefully designed prompt strategy to guide the Mistral model through the research process. The prompts include:

1. **Base Template**: Defines the research area and constraints
2. **Memory Optimization**: Carefully manages context window usage
3. **Output Formatting**: Ensures results are in parseable JSON format
4. **Novelty Assessment**: Helps evaluate the uniqueness of ideas

## Future Development

Planned enhancements include:

1. **Complete Experiment Phase**: Run experiments with our Mistral model
2. **Complete Writeup Phase**: Generate research papers based on experiments
3. **Enhanced Templates**: More diverse research templates
4. **Integration with Research Tools**: Connect to external tools and libraries 