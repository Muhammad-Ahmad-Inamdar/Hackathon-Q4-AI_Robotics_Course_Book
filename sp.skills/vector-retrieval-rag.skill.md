# Vector Data Retrieval for RAG Chatbot

## Skill Name
Vector Data Retrieval for RAG Chatbot

## Problem it Solves
Enables accurate and efficient similarity search in vector databases for RAG (Retrieval-Augmented Generation) chatbot systems. Specifically addresses issues with vector similarity search using Qdrant, metadata filtering, and fallback mechanisms when strict filters return zero results.

## Common Mistakes (based on past failures)
- Not implementing proper metadata filtering for chapter/page boundaries
- Failing to handle cases where strict filters return zero results
- Incorrect vector dimension matching between query and stored vectors
- Not implementing proper fallback logic when initial searches fail
- Improper handling of similarity thresholds
- Not accounting for metadata inconsistencies in retrieved results

## Proven Working Pattern (Golden Path)
- Use Qdrant vector database with proper collection schema
- Implement multi-level filtering (strict first, then relaxed)
- Apply metadata filtering for chapter and page boundaries
- Implement fallback to broader search when strict filters yield no results
- Use cosine similarity with appropriate threshold values
- Include proper error handling for vector operations

## Guardrails (what must NEVER be changed)
- Vector dimension consistency between embeddings and database schema
- Metadata field names and types in the vector database
- Minimum similarity threshold values (never set below 0.3)
- Fallback logic sequence (strict → relaxed → broader search)
- Error handling for empty result sets

## Step-by-Step Execution
1. Initialize Qdrant client connection with proper configuration
2. Generate embedding for user query using the same model as stored vectors
3. Prepare search parameters with strict filters (chapter, page, etc.)
4. Execute initial search with strict filters
5. If no results found, execute fallback search with relaxed filters
6. Apply similarity threshold filtering to results
7. Validate and return metadata along with content
8. Handle edge cases (empty results, connection failures)

## Verification Checklist
- [ ] Vector dimensions match between query and stored embeddings
- [ ] Metadata filtering correctly applied (chapter, page boundaries)
- [ ] Fallback logic triggers when strict filters return zero results
- [ ] Similarity threshold properly implemented (0.3 minimum)
- [ ] Error handling covers empty result sets
- [ ] Qdrant client connection stable and properly configured
- [ ] Search performance within acceptable time limits (<2 seconds)

## Reusability Notes
This pattern can be adapted to other vector databases (Pinecone, Weaviate) by changing the client initialization and search syntax. The fallback logic and metadata filtering principles remain consistent across different vector database implementations. Ensure the embedding model remains consistent between indexing and querying phases.