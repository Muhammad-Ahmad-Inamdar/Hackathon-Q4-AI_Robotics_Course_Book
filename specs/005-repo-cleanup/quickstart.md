# Quickstart: Repository Cleanup Implementation

## Overview

This guide provides a step-by-step approach to implementing the repository cleanup feature, ensuring safe removal of unnecessary files while preserving critical functionality.

## Prerequisites

- Git installed and configured
- Node.js and npm (for Docusaurus verification)
- Python 3.11+ (for any remaining backend verification)
- Basic understanding of the repository structure

## Step-by-Step Implementation

### Phase 1: Repository Audit

1. **Create a backup branch**
   ```bash
   git checkout -b backup-before-cleanup
   git push origin backup-before-cleanup
   ```

2. **Analyze current repository structure**
   ```bash
   # Get initial repository size
   du -sh .

   # List all files and directories
   find . -type f | wc -l  # Count of files
   find . -type d | wc -l  # Count of directories
   ```

3. **Document current functionality**
   - Test that the site builds: `npm run build`
   - Verify chatbot UI loads: `npm start`
   - Check that documentation renders correctly

### Phase 2: File Classification

1. **Identify RAG & Vector Backend components to remove**
   - `backend/src/rag_agent/` directory and all contents
   - Any Qdrant-related files
   - Vector database clients and utilities
   - Backend API files that support RAG functionality

2. **Identify testing/experimental artifacts to remove**
   - `backend/tests/` directory
   - Temporary scripts and utilities
   - Proof-of-concept implementations
   - Debug scripts

3. **Identify unnecessary files to remove**
   - Empty directories
   - Duplicate files
   - Unused configuration files
   - Generated artifacts not needed at runtime

4. **Identify files to preserve**
   - All frontend components in `src/`
   - Documentation in `docs/`
   - Static assets in `static/`
   - Build configuration files

### Phase 3: Chatbot UI Safety Implementation

1. **Locate backend API dependencies in chatbot**
   - Check `src/components/Chatbot/api/chatService.js`
   - Identify all API endpoints the chatbot uses

2. **Replace backend calls with mock implementations**
   ```javascript
   // Example mock implementation
   export const sendMessage = async (message) => {
     return new Promise((resolve) => {
       setTimeout(() => {
         resolve({
           success: true,
           data: {
             response: "Backend under maintenance. This is a mock response."
           }
         });
       }, 500);
     });
   };
   ```

3. **Test UI resilience**
   - Ensure UI doesn't crash when backend is unavailable
   - Verify graceful degradation messages are displayed

### Phase 4: Cleanup Execution

1. **Remove classified files**
   ```bash
   # Remove RAG backend components
   rm -rf backend/src/rag_agent/

   # Remove test directories
   rm -rf backend/tests/

   # Remove other classified files/directories
   # (Based on your classification results)
   ```

2. **Verify functionality after each removal batch**
   - Run `npm run build` to ensure site still builds
   - Run `npm start` to ensure UI loads properly

3. **Update any broken imports/references**
   - Search for imports that referenced removed files
   - Update or remove references as appropriate

### Phase 5: Verification and Reporting

1. **Run comprehensive tests**
   - Build the site: `npm run build`
   - Start the development server: `npm start`
   - Verify all documentation pages render correctly
   - Test chatbot UI functionality

2. **Measure cleanup results**
   ```bash
   # Check final repository size
   du -sh .

   # Compare with initial size
   echo "Size reduction achieved"
   ```

3. **Create cleanup report**
   - Document all removed files/directories
   - Categorize by type (RAG backend, tests, etc.)
   - Note any files that were reviewed but preserved
   - Include verification results

## Verification Checklist

- [ ] Site builds successfully after cleanup (`npm run build`)
- [ ] Development server starts without errors (`npm start`)
- [ ] All documentation pages render correctly
- [ ] Chatbot UI loads without crashing
- [ ] Chatbot UI handles missing backend gracefully
- [ ] Repository size reduced by at least 30%
- [ ] All essential functionality preserved
- [ ] Cleanup report created with all required details

## Rollback Plan

If issues arise during cleanup:

1. Switch back to the backup branch:
   ```bash
   git checkout backup-before-cleanup
   ```

2. Or revert to the last known good commit:
   ```bash
   git reset --hard <commit-hash>
   ```