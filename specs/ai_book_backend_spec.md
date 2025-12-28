# AI Book Backend Specification

## Project Overview
The AI Book Backend is a comprehensive system designed to ingest, process, and store content from websites for AI-powered search and retrieval. The system focuses on the frontend URL: https://muhammad-ahmad-inamdar.github.io/Hackathon-Q4-AI_Robotics_Course_Book/

## Functional Requirements

### 1. Website Ingestion
- Extract all URLs from a given base URL up to a maximum number of pages
- Prioritize sitemap.xml for efficient URL discovery
- Fall back to link crawling if sitemap is not available
- Respect domain boundaries to avoid crawling external sites
- Handle various types of web content gracefully

### 2. Text Processing
- Extract clean text content from web pages
- Remove HTML tags, scripts, and styling elements
- Preserve meaningful content while discarding noise

### 3. Text Chunking
- Implement deterministic chunking algorithm
- Configurable chunk size (default: 512 tokens)
- Configurable overlap (default: 50 tokens)
- Maintain context across chunk boundaries

### 4. Embedding Generation
- Use Cohere's multilingual embedding model
- Generate high-quality vector representations
- Handle API rate limits and errors

### 5. Vector Storage
- Store embeddings in Qdrant Cloud
- Include metadata for each chunk
- Enable efficient similarity search

## Technical Specifications

### System Design Functions
1. `get_all_urls(base_url, max_pages)`: Extract all URLs from website
2. `extract_text_from_url(url)`: Extract clean text from URL
3. `chunk_text(text)`: Split text into chunks with overlap
4. `embed(texts)`: Generate embeddings using Cohere
5. `create_collection()`: Create Qdrant collection
6. `save_chunk_to_qdrant(chunk_data, embedding, url)`: Store chunk in Qdrant
7. `ingest_book(base_url)`: Main orchestration function
8. `main()`: Entry point execution

### Configuration Parameters
- COHERE_API_KEY: API key for Cohere services
- QDRANT_API_KEY: API key for Qdrant Cloud
- QDRANT_HOST: Host URL for Qdrant Cloud instance
- CHUNK_SIZE: Size of text chunks in tokens (default: 512)
- CHUNK_OVERLAP: Overlap between chunks in tokens (default: 50)
- VECTOR_COLLECTION_NAME: Name of Qdrant collection

### External Dependencies
- Cohere API for embedding generation
- Qdrant Cloud for vector storage
- Web scraping libraries for content extraction

## Performance Requirements
- Efficient crawling without overwhelming target servers
- Proper handling of API rate limits
- Memory-efficient processing of large documents
- Fast vector storage and retrieval

## Security Considerations
- Secure storage of API credentials
- Input validation for URLs
- Rate limiting to respect target servers
- Proper error handling to prevent information disclosure

## Quality Assurance
- Comprehensive error handling
- Detailed logging for debugging
- Graceful degradation when services are unavailable
- Validation of embeddings before storage