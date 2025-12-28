# AI Book Backend Implementation Plan

## Overview
This plan outlines the implementation of a backend system for the AI Book project that handles website ingestion, text processing, embedding generation, and vector storage using Cohere and Qdrant.

## Objectives
- Create a robust backend service that can ingest content from websites
- Process and chunk text content deterministically
- Generate embeddings using Cohere models
- Store embeddings and metadata in Qdrant Cloud for efficient retrieval
- Provide a unified system in a single main.py file for easy debugging

## Technical Requirements

### 1. Project Setup
- Initialize Python project using UV package management
- Create proper directory structure
- Set up dependencies for web scraping, embeddings, and vector storage

### 2. Environment Configuration
- Configure environment variables for Cohere API
- Configure environment variables for Qdrant Cloud
- Secure credential management

### 3. Core Functionality
- URL extraction and website crawling capability
- Text extraction from web pages
- Deterministic text chunking algorithm
- Embedding generation using Cohere
- Vector storage in Qdrant Cloud

## Implementation Phases

### Phase 1: Project Initialization
- Create backend directory
- Initialize Python project with UV
- Set up project dependencies
- Configure basic project structure

### Phase 2: Configuration Setup
- Create .env file with Cohere and Qdrant credentials
- Set up configuration management
- Implement environment variable loading

### Phase 3: Core Function Implementation
- Implement `get_all_urls` function for website crawling
- Implement `extract_text_from_url` for content extraction
- Implement `chunk_text` for deterministic text chunking
- Implement `embed` for generating embeddings with Cohere
- Implement `create_collection` for Qdrant setup
- Implement `save_chunk_to_qdrant` for vector storage
- Implement `ingest_book` as main orchestration function

### Phase 4: Integration and Testing
- Integrate all functions in a single main.py file
- Implement proper error handling
- Add comprehensive logging
- Test with the frontend URL: https://muhammad-ahmad-inamdar.github.io/Hackathon-Q4-AI_Robotics_Course_Book/

## System Architecture

### Components
1. **Sitemap Reader**: Extracts URLs from sitemap.xml (primary method)
2. **Website Crawler**: Extracts URLs by crawling links (fallback method)
3. **Text Extractor**: Extracts clean text content from web pages
4. **Chunker**: Splits text into manageable chunks with overlap
5. **Embedder**: Generates embeddings using Cohere API
6. **Vector Storage**: Stores embeddings in Qdrant Cloud
7. **Orchestrator**: Coordinates the entire ingestion process


### Data Flow
1. Input: Base URL of the book/website
2. URL Discovery: Check for sitemap.xml first, fall back to link crawling
3. Extraction: Get text content from each URL
4. Processing: Chunk text into manageable pieces
5. Embedding: Generate vector representations
6. Storage: Save vectors and metadata to Qdrant
7. Output: Ready for AI-powered search and retrieval

## Success Criteria
- All functions implemented in a single main.py file
- Successful ingestion of the target website
- Proper storage of embeddings in Qdrant
- Error handling for network issues and API failures
- Comprehensive logging for debugging

## Dependencies
- Cohere Python SDK
- Qdrant Python Client
- Beautiful Soup for web scraping
- Requests for HTTP operations
- Python-dotenv for environment management
- Tiktoken for token counting