# Research: Repository Cleanup and Maintenance

## Overview

This document captures research findings for the repository cleanup feature, addressing unknowns and technology choices identified during the planning phase.

## Technology Context Research

### Backend Technologies Identified

- **Python Backend**: The repository contains a Python-based RAG agent system in `backend/src/rag_agent/`
- **RAG Components**: Includes vector database client (likely Qdrant), embedding generation, and retrieval pipeline
- **Frontend**: React/Docusaurus based system with chatbot UI components
- **Build System**: Docusaurus-based documentation and static site generation

## File Classification Strategy

### KEEP Category
Files and directories that must be preserved based on functional requirements:

- **Frontend application**: All files in `src/` directory
- **Chatbot UI components**: `src/components/Chatbot/` and related files
- **Docusaurus setup**: `docusaurus.config.js`, `sidebars.js`, `package.json`
- **Documentation/book files**: `docs/` directory
- **UI assets and static files**: `static/`, `img/`, CSS files
- **Core build configuration**: `package.json`, `docusaurus.config.js`

### REMOVE Category
Files and directories safe to remove based on functional requirements:

- **RAG & Vector Backend**:
  - `backend/src/rag_agent/` directory and all subfiles
  - `backend/main.py`, `backend/requirements.txt`, `backend/pyproject.toml` (backend-only)
  - Qdrant client and tool files: `backend/src/rag_agent/qdrant_client.py`, `qdrant_tool.py`
  - Vector utilities: `backend/src/rag_agent/vector_utils.py`
  - Embedding generation code
- **Testing artifacts**: `backend/tests/` directory
- **Experimental files**: Any temporary or debug scripts
- **Generated artifacts**: Build outputs, cache files

### REVIEW Category
Files requiring careful review before removal:

- **Configuration files** that might be shared between frontend/backend
- **API endpoints** that chatbot UI might depend on
- **Environment files** that might affect frontend operation

## Chatbot UI Safety Requirements

### Backend API Dependencies
Research shows that the chatbot UI in `src/components/Chatbot/` likely depends on backend APIs that will be removed during cleanup.

**Required Action**: Replace actual backend API calls with safe stubs or mock responses to prevent UI crashes.

### Implementation Approach
- Identify all API calls in `src/components/Chatbot/api/chatService.js`
- Create mock implementations that return placeholder responses
- Ensure UI gracefully handles "backend under maintenance" scenarios
- Maintain existing UI interface contracts to prevent breaking changes

## Risk Mitigation

### Verification Strategy
- Create comprehensive backup before starting cleanup
- Implement cleanup in phases with verification checkpoints
- Test build process after each phase
- Test chatbot UI functionality after backend removal
- Validate documentation rendering

### Rollback Plan
- Git-based rollback using pre-cleanup commit
- File system backup of critical directories before removal
- Gradual removal with frequent commits to enable targeted rollbacks

## Tools and Scripts for Cleanup

### Classification Process
- Use file analysis tools to identify dependencies
- Cross-reference imports/exports to identify truly unused files
- Verify build and run commands after each cleanup phase

### Automation Potential
- Bash/PowerShell scripts to automate bulk file operations
- Git-based verification to ensure no critical files are accidentally removed
- Automated testing to verify functionality after cleanup