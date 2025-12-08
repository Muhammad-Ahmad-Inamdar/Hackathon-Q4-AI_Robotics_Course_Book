# ADR-3: RAG Chatbot Architecture for Textbook Q&A System

**Status**: Accepted
**Date**: 2025-12-07

## Context

The textbook needs an intelligent Q&A system that can answer questions based on the textbook content. This requires implementing a Retrieval-Augmented Generation (RAG) system that can understand user queries and provide accurate answers based on the textbook modules.

## Decision

We will implement a RAG system with the following architecture:

- **Backend**: FastAPI application handling chat requests
- **Embeddings**: Text embedding using OpenAI or compatible embedding models
- **Vector Storage**: Qdrant Cloud Free Tier for vector similarity search
- **Database**: Neon Serverless Postgres for metadata and conversation history
- **AI Integration**: OpenAI Agents/ChatKit SDK for conversation management
- **Integration**: Embedded in Docusaurus frontend via API calls

The RAG pipeline will:
1. Receive user query from frontend
2. Embed the query using text embedding model
3. Retrieve relevant textbook content from vector store
4. Generate response using LLM with retrieved context
5. Return answer to frontend with source references

## Alternatives Considered

1. **Different RAG approach**: Simple keyword search instead of vector embeddings
   - Pros: Simpler to implement, potentially faster
   - Cons: Less semantic understanding, lower quality responses

2. **Different vector database**: Pinecone, Weaviate, or self-hosted instead of Qdrant
   - Pros: Different features, pricing models, or performance characteristics
   - Cons: Learning curve for new technology, potential vendor lock-in

3. **Different architecture**: Client-side processing vs server-side
   - Pros: Potentially lower latency, reduced server load
   - Cons: Security concerns, larger client bundle, less control over processing

## Consequences

**Positive**:
- Students can get immediate answers to questions about textbook content
- Semantic search provides more relevant results than keyword matching
- Scalable architecture that can handle increasing content
- Proper source attribution for generated answers

**Negative**:
- More complex architecture with multiple services
- Potential latency for real-time responses
- Costs associated with vector database and AI services
- Complexity of ensuring accuracy of generated responses

## References

- spec.md: Functional Requirements (FR-008) and Non-Functional Requirements (NFR-009, NFR-010)
- plan.md: Phase 2: Integration & Backend section