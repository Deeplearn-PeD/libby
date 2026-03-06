# Changelog

All notable changes to Libby D. Bot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-03-06

### Added
- Integrated Ollama server directly in Docker container
- Pre-installed `mxbai-embed-large` embedding model in Docker image
- Added `docker/start.sh` startup script for automatic Ollama initialization
- Added `ollama-models` volume for persisting downloaded models
- Added `zstd` package for Ollama installation support

### Changed
- Updated Dockerfile to include Ollama server installation
- Updated docker-compose.yml to use internal Ollama instead of host connection
- Increased health check start period to 60s for Ollama initialization
- Updated documentation with new Docker deployment instructions
- Improved Docker build process with model pre-loading

### Fixed
- Fixed Docker connectivity issues by embedding Ollama server in container
- Fixed port conflicts by not exposing Ollama port externally (11434)

## [0.8.0] - 2026-03-03

### Added
- REST API server with FastAPI for programmatic access
- Docker support with Dockerfile and docker-compose configuration
- Re-embedding functionality to update embeddings with new models
- `libby-server` CLI command to start the API server
- API endpoints for text embedding, file upload, document retrieval, and health checks
- `.env.example` file for environment configuration

### Changed
- Bumped version from 0.7.0 to 0.8.0
- Made server more robust with better error handling
- Updated documentation with API usage examples
- Improved configuration management

### Fixed
- Fixed various bugs in the API server
- Fixed server stability issues

## [0.7.0] - 2026-02-13

### Added
- RAG (Retrieval Augmented Generation) capabilities
- Document embedder class for managing embeddings
- Library search tool for agents
- Support for DuckDB with HNSW indexes
- New database files for embedding and memory storage
- PostgreSQL backend tests

### Changed
- Refined embedding logic and improved tests
- Updated dependencies
- Locked project dependencies using uv

### Fixed
- Corrected DuckDB table creation (replaced `SERIAL` with `nextval`)
- Fixed DuckDB connection initialization
- Fixed connection handling to properly use `self.engine`
- Corrected AUTOINCREMENT syntax for DuckDB
- Improved DuckDB interaction using direct SQL queries

### Removed
- Removed `EmbeddingDuckdb` class, simplified table creation

## [0.6.0] - 2025-12-08

### Added
- Full DuckDB support with HNSW vector indexes
- Hybrid search combining vector and full-text search (FTS)
- FTS index creation for DuckDB

### Changed
- Migrated from SQLAlchemy to direct SQL for DuckDB
- Improved database connection management
- Updated code structure and fixed multiple bugs

### Fixed
- Fixed many bugs related to database connections
- Corrected table creation and query execution
- Fixed SQLite fallback in DocEmbedder

## [0.5.0] - 2025-08-17

### Added
- SQLite support with sqlite-vec extension for embeddings
- Method to retrieve embedded documents (`get_embedded_documents`)
- Connection property to manage database connections
- Multi-thread mode for SQLite

### Changed
- Migrated embedding storage from SQLAlchemy to direct SQLite
- Improved database configuration and embedding handling
- Changed default embedding model to Gemini
- Refined tests and improved test coverage

### Fixed
- Fixed SQLite3 database closing issues
- Corrected table creation and queries for SQLite with vec0
- Fixed fallback behavior for SQLite in DocEmbedder

## [0.4.0] - 2025-08-08

### Added
- Gemini embedding model support
- Google as default AI provider
- Support for multiple embedding models (Gemini, Ollama)

### Changed
- Updated settings configuration to be more robust
- Improved provider handling and configuration
- Bumped version from 0.3.x to 0.4.0

### Fixed
- Fixed bugs in embedding generation
- Corrected settings loading

## [0.3.11] - 2025-06-25

### Fixed
- Adapted `ask` function to remove `<think\>` content from reasoning models' responses

## [0.3.10] - 2025-05-10

### Changed
- Updated dependencies
- Updated documentation

## [0.3.9] - 2025-03-02

### Added
- API reference documentation
- Comprehensive tutorial documentation

### Changed
- Migrated configuration from YAML to Pydantic settings
- Updated documentation structure
- Removed PyYAML dependency

### Removed
- Removed YAML configuration files

## [0.3.8] - 2025-01-04

### Added
- Method to embed a path (directory of documents)
- Initial support for SQLite and DuckDB as backends for DocEmbedder
- Flexible database URL support

### Changed
- Updated package dependencies
- Improved database URL handling

## [0.3.7] - 2024-11-20

### Added
- Qwen model support
- Support for reading prompts from input text files
- Context support for the generate command
- Option to specify LLM model on CLI
- Model selection from config with validation

### Changed
- Increased number of retrieved documents from 15 to 100
- Updated README with comprehensive usage instructions
- Bumped version to 0.3.x series

### Fixed
- Fixed bug in calling generate with a prompt file

## [0.3.0] - 2024-07-31

### Added
- Ingest module with PDF processing capabilities
- Tests for CLI functionality
- Support for collections of embeddings
- Filter by collection when retrieving documents

### Changed
- Improved documentation
- Refactored CLI for better testability
- Bumped version from 0.2.x to 0.3.0

## [0.2.0] - 2024-07-05

### Added
- Memory functionality for Libby
- Chat history storage and retrieval
- Support for multiple chat sessions

### Changed
- Improved memory implementation
- Enhanced conversation context handling

## [0.1.0] - 2024-04-09

### Added
- Initial release of Libby D. Bot
- Basic CLI interface with Fire
- Support for multiple AI models (Llama3, Gemma, ChatGPT)
- PDF document processing and embedding
- Question answering with context
- Content generation capabilities
- Poetry configuration and dependencies
- Basic project structure

### Features
- Multiple language support (English and Portuguese)
- Various AI models available
- PDF document processing
- Question answering with document context
- Content generation

[0.9.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.3.11...v0.4.0
[0.3.11]: https://github.com/Deeplearn-PeD/libby/compare/v0.3.10...v0.3.11
[0.3.10]: https://github.com/Deeplearn-PeD/libby/compare/v0.3.9...v0.3.10
[0.3.9]: https://github.com/Deeplearn-PeD/libby/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/Deeplearn-PeD/libby/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/Deeplearn-PeD/libby/compare/v0.3.0...v0.3.7
[0.3.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Deeplearn-PeD/libby/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Deeplearn-PeD/libby/releases/tag/v0.1.0
