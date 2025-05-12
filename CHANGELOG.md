# Changelog

All notable changes to the Veritas project will be documented in this file.

## [1.2.0] - 2025-05-20

### Added
- AI Scientist: A new core component for research assistance
- Comprehensive testing suite for the AI Scientist
- Memory optimization for AI Scientist on M4 Mac
- Test runner script (test_all.sh) for systematic testing

### Changed
- Reorganized directory structure: moved from ai_scientist_integration to ai_scientist
- Enhanced documentation with detailed usage examples
- Improved error handling in the research generation pipeline
- Updated class and method names for consistency

### Fixed
- Memory leaks during research idea generation
- Inconsistent output formatting in JSON responses
- Path resolution issues in the template loading

## [1.1.0] - 2025-05-11

### Added
- Terminal UI with three-part response display (context, direct answer, combined)
- Improved context chunking for better memory efficiency
- Support for M4 Max optimization with up to 128GB RAM
- Memory clean-up routines to prevent OOM errors

### Changed
- Optimized RAG implementation for better performance
- Reduced default `max_new_tokens` from 512 to 200 for stability
- Changed default `max_retrieved_chunks` from 5 to 2 for memory efficiency
- Updated environment variables for better MPS memory management
- Improved prompt format for better context handling

### Fixed
- Fixed OOM errors with MPS backend by setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- Fixed context processing to prevent nonsensical responses
- Fixed RAM usage spikes during generation
- Addressed kernel_task high CPU usage on Apple Silicon

### Removed
- Removed quantization fallback logic in favor of direct full precision loading
- Removed multiprocessing worker implementation in favor of simpler architecture

## [1.0.0] - 2025-04-15

### Added
- Initial RAG system implementation
- Support for Mistral 2 7B model
- Document processing pipeline
- FAISS indexing for efficient retrieval
- Basic CLI interface
- Support for CUDA and MPS acceleration 