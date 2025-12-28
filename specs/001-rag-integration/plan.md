# Implementation Plan: RAG System Integration & Cleanup

**Branch**: `001-rag-integration` | **Date**: 2025-12-26 | **Spec**: [link]
**Input**: Feature specification from `/specs/001-rag-integration/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement RAG system integration with Gemini 2.0 Flash-Lite API, integrate with Docusaurus frontend, implement reasoning engine for indirect queries, and perform project cleanup. The system will enable users to ask questions about the "Physical AI & Humanoid Robotics" textbook and receive accurate answers through a chatbot interface. Implementation will follow the 5-phase approach specified by the user.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/Node.js for frontend
**Primary Dependencies**: FastAPI, Qdrant, Google Generative AI SDK, Docusaurus, React
**Storage**: Qdrant vector database for embeddings, textbook content files
**Testing**: pytest for backend, Jest for frontend
**Target Platform**: Linux server deployment, Web browser compatible
**Project Type**: Web application (frontend + backend)
**Performance Goals**: Response time under 10 seconds for 95% of queries
**Constraints**: Must handle textbook content analysis, support "Thinking..." UI states, CORS enabled
**Scale/Scope**: Single textbook Q&A system supporting multiple concurrent users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Accuracy First: System must ground responses strictly in textbook context
- Architectural Integrity: Maintain separation between Docusaurus frontend and FastAPI backend
- Zero-Clutter Policy: Remove unnecessary files during cleanup phase
- Security: API keys must not be committed to version control
- API Engine Standards: Use Gemini 2.0 Flash-Lite (v1 API) as specified
- Reasoning Protocols: Implement "Step-Back Reasoning" for indirect queries

## Implementation Phases

### Phase 1 (Cleanup):
- Scan root directory for unused files (.txt logs, manual ingestion scripts, duplicate .md files)
- Propose a list of files to delete for user approval
- Execute cleanup to ensure clean root directory structure
- Ensure compliance with Zero-Clutter Policy from constitution

### Phase 2 (Credential Sync):
- Copy relevant `.env` variables from the source directory
- Create an `.env.example` for the repo
- Ensure sensitive information is not committed to version control
- Maintain security requirements from constitution

### Phase 3 (Logic Migration):
- Refactor the current API code to match the "Successfull" project's RAG and embedding flow
- Add logical reasoning instructions to the Gemini prompt using Step-Back Reasoning
- Ensure integration with existing Qdrant vector database
- Maintain API Engine Standards from constitution

### Phase 4 (Frontend Bridge):
- Modify the React/Docusaurus chatbot component to send POST requests to the local/live API
- Implement "Thinking..." states in UI to provide feedback during query processing
- Ensure proper CORS handling between frontend and backend
- Maintain architectural integrity between frontend and backend

### Phase 5 (Validation):
- Run test queries including "How many chapters are in this book?" to verify reasoning engine
- Verify source count and answer quality against textbook content
- Ensure system responds within 10 seconds for 95% of queries
- Validate Docusaurus site remains responsive with zero console errors

## Project Structure

### Documentation (this feature)
```text
specs/001-rag-integration/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)
```text
backend/
├── src/
│   ├── rag_agent/
│   │   ├── agent.py
│   │   ├── qdrant_tool.py
│   │   ├── vector_utils.py
│   │   └── api/
│   │       ├── main.py
│   │       └── endpoints.py
└── tests/

src/
├── components/
│   ├── Chatbot/
│   │   ├── Chatbot.jsx
│   │   ├── ChatWindow.js
│   │   └── api/
│   │       └── chatService.js
└── pages/
```

**Structure Decision**: Web application with separate backend API and Docusaurus frontend, following architectural integrity principle from constitution. Leverage existing RAG components in backend/src/rag_agent and integrate with frontend Chatbot components.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |