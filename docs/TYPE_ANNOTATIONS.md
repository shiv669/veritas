# Type Annotation Standards for Veritas

This document outlines the standards for type annotations throughout the Veritas codebase. Consistent type annotations improve code quality, enable better static analysis, and provide clearer documentation.

## General Principles

1. **All public functions and methods must have type annotations** for parameters and return values.
2. **Use the centralized type definitions** in `src/veritas/typing.py` whenever possible.
3. **Be as specific as possible** with types, avoiding `Any` when a more specific type can be used.
4. **Document complex types** with comments or docstrings.
5. **Use consistent import style** for typing modules.

## Import Style

```python
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
import numpy as np
from src.veritas.typing import ChunkType, ChunkList, EmbeddingType
```

## Basic Type Annotations

### Function Annotations

```python
def process_document(text: str, chunk_size: int = 512) -> List[str]:
    """Process a document into chunks."""
    # Implementation...
```

### Class Annotations

```python
class DocumentProcessor:
    """Process documents for the RAG system."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process(self, text: str) -> List[str]:
        """Process text into chunks."""
        # Implementation...
```

### Variable Annotations

```python
# Module-level variables
DEFAULT_CHUNK_SIZE: int = 512
MODEL_PATHS: Dict[str, str] = {
    "mistral": "models/mistral-7b",
    "embedding": "sentence-transformers/all-MiniLM-L6-v2"
}

# Function-level variables
def process_chunks(chunks: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    count: int = 0
    # Implementation...
```

## Complex Type Annotations

### Optional Parameters

```python
def load_model(model_path: str, device: Optional[str] = None) -> Any:
    """Load a model from path."""
    if device is None:
        device = get_default_device()
    # Implementation...
```

### Union Types

```python
def parse_input(input_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[str]:
    """Parse various input formats into a list of strings."""
    # Implementation...
```

### Callable Types

```python
def apply_function(items: List[Any], func: Callable[[Any], Any]) -> List[Any]:
    """Apply a function to each item in a list."""
    return [func(item) for item in items]
```

### TypedDict for Structured Dictionaries

```python
from typing import TypedDict, List

class ChunkDict(TypedDict):
    text: str
    source: str
    page: int
    embedding: List[float]

def process_chunk(chunk: str) -> ChunkDict:
    """Process a chunk into a structured dictionary."""
    # Implementation...
```

## Type Annotations with Generic Types

```python
from typing import TypeVar, Generic, List

T = TypeVar('T')

class Queue(Generic[T]):
    def __init__(self):
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop(0)
```

## Type Checking with mypy

We use `mypy` for static type checking. Run the following command to check types:

```bash
mypy src/
```

Configuration for mypy is in `.mypy.ini` at the project root.

## Common Type Definitions

Refer to `src/veritas/typing.py` for common type definitions used throughout the project, including:

- `ChunkType` - Type for document chunks
- `EmbeddingType` - Type for vector embeddings
- `MessageList` - Type for chat message lists
- `ResearchIdea` - Type for AI Scientist research ideas

## Adding New Types

When adding new types:

1. If the type is used in multiple modules, add it to `src/veritas/typing.py`
2. If the type is specific to one module, define it at the top of that module
3. Document the type with a comment or docstring
4. Use consistent naming conventions (e.g., `TypeName` for classes, `TypeName` for aliases) 