# Research: RAG System Integration & Cleanup

## Decision: Selected Architecture Pattern
**What was chosen**: Web application with separate FastAPI backend and Docusaurus frontend
**Rationale**: Maintains architectural integrity as required by project constitution, allows for proper separation of concerns, and enables independent scaling of components.

## Decision: AI Provider and Model
**What was chosen**: Google Generative AI with Gemini 2.0 Flash-Lite model
**Rationale**: Explicitly required by project constitution (API Engine Standards principle), provides optimal balance of speed and accuracy for Q&A functionality.

## Decision: Vector Database
**What was chosen**: Qdrant vector database
**Rationale**: Already present in the codebase (backend/src/rag_agent/qdrant_client.py), proven integration with existing RAG components, efficient for textbook content retrieval.

## Decision: Reasoning Engine Implementation
**What was chosen**: Step-Back Reasoning pattern for handling indirect queries
**Rationale**: Required by project constitution (Reasoning Protocols principle), enables handling of queries like "How many chapters?" by breaking them down into sub-questions.

## Decision: Frontend Integration Approach
**What was chosen**: POST requests from Docusaurus React components to backend API endpoints
**Rationale**: Standard web application pattern, enables proper CORS handling as required by functional requirements, maintains clean separation between frontend and backend.

## Decision: Cleanup Scope
**What was chosen**: Remove temporary files, logs, duplicate documentation, and unused configuration files
**Rationale**: Aligns with Zero-Clutter Policy from project constitution, improves maintainability and repository hygiene.

## Alternatives Considered

### AI Provider Alternatives
- OpenAI GPT models: Rejected due to constitution requirement for Gemini 2.0 Flash-Lite
- Open-source models: Rejected due to performance requirements and existing integration patterns
- Other Google models: Rejected as Gemini 2.0 Flash-Lite specifically required by constitution

### Vector Database Alternatives
- Pinecone: Rejected due to existing Qdrant integration in codebase
- Chroma: Rejected due to existing Qdrant integration in codebase
- PostgreSQL with pgvector: Rejected due to specialized vector search requirements

### Frontend Integration Alternatives
- WebSocket communication: Rejected due to complexity and HTTP REST being sufficient for use case
- GraphQL: Rejected due to simplicity requirements and existing REST patterns in codebase