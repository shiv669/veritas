# Veritas Project Structure

This document outlines the organized structure of the Veritas repository after reorganization.

## Root Directory

The root directory now contains only essential project files:

```
veritas/
├── README.md              # Main project documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation configuration
├── install.sh            # Convenience installation script
├── .gitignore            # Git ignore patterns
└── directories/          # See directory structure below
```

## Directory Structure

### `/src/` - Source Code
Contains the main Veritas package implementation:
- `veritas/` - Main package directory
  - `ai_scientist/` - AI Scientist research capabilities
  - Core modules (rag.py, config.py, etc.)

### `/scripts/` - Executable Scripts
Contains operational scripts for running the system:
- Data processing pipeline scripts (1-7)
- `run.py` - Main unified interface
- `cli.py` - Command-line interface
- Indexing and retrieval utilities

### `/docs/` - Documentation
All documentation files organized together:
- `CHANGELOG.md` - Version history (moved from root)
- `ARCHITECTURE.md` - System architecture
- `AI_SCIENTIST.md` - AI Scientist documentation
- `GETTING_STARTED.md` - Getting started guide
- `TYPE_ANNOTATIONS.md` - Type system documentation
- `quickstart.md` - Quick start guide
- `veritas-ai-scientist.gif` - Demo animation

### `/tools/` - Utilities & Tools
Installation and development utilities:
- `install.py` - Unified installation script (moved from root)

### `/config/` - Configuration Files
Configuration files organized separately:
- `.mypy.ini` - MyPy type checker configuration (moved from root)

### `/models/` - Model Storage
Directory for storing downloaded models and indices

### `/results/` - Output Results
Directory for storing experiment results and outputs

## Benefits of This Organization

1. **Cleaner Root**: Only essential files remain in the root directory
2. **Logical Grouping**: Related files are grouped in appropriate directories
3. **Maintainability**: Easier to find and maintain configuration files
4. **Scalability**: Better structure for future growth
5. **Convenience**: Install script still easily accessible via `./install.sh`

## Migration Notes

- All references to moved files have been updated
- A convenience script (`install.sh`) maintains easy access to installation
- Documentation references have been updated to reflect new locations
- The project maintains backward compatibility for all user-facing interfaces 