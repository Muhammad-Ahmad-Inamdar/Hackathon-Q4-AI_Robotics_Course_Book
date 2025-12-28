# Complete RAG Chatbot Development & Working Flow

## Skill Name
Complete RAG Chatbot Development & Working Flow

## Problem it Solves
Establishes a complete, stable workflow for developing and operating a RAG (Retrieval-Augmented Generation) chatbot system that combines vector search, LLM integration, and conversation management. Addresses common integration failures between components and ensures reliable user interactions.

## Common Mistakes (based on past failures)
- Not implementing proper fallback when vector search returns no results
- Improper context window management leading to LLM token overflow
- Not handling concurrent user sessions properly
- Failing to validate retrieved context quality before LLM processing
- Not implementing proper error boundaries between system components
- Missing proper conversation history management and context injection
- Inadequate prompt injection protection

## Proven Working Pattern (Golden Path)
- Implement robust vector search with fallback mechanisms
- Use sliding window for conversation history management
- Validate retrieved context quality before LLM processing
- Implement proper error handling between all system components
- Apply rate limiting and resource management
- Use structured prompt templates with clear role separation
- Implement proper session state management

## Guardrails (what must NEVER be changed)
- Vector search must always have fallback to broader search
- Context length must never exceed LLM token limits (leave 10% buffer)
- User input must always be sanitized before processing
- System prompts must be kept separate from user content
- Session isolation between different users must be maintained
- Error messages must never expose system internals to users

## Step-by-Step Execution
1. Receive and sanitize user input
2. Generate embedding for user query
3. Execute vector search with strict filters first
4. Apply fallback search if initial results are empty
5. Validate and rank retrieved context quality
6. Format conversation history with proper context injection
7. Construct LLM prompt with retrieved context and conversation history
8. Call LLM with proper timeout and error handling
9. Process and format LLM response
10. Update conversation history with new interaction
11. Return response to user with proper error handling

## Verification Checklist
- [ ] Vector search fallback triggers when initial search returns no results
- [ ] Context length stays within LLM token limits with 10% buffer
- [ ] Conversation history properly maintained and limited
- [ ] User input sanitized to prevent injection attacks
- [ ] Error handling covers all system components
- [ ] Session state properly isolated between users
- [ ] Response times remain within acceptable limits (<5 seconds)
- [ ] Context quality validation prevents poor information injection

## Reusability Notes
This pattern can be adapted to different LLM providers (OpenAI, Anthropic, Ollama) by changing the LLM call interface. The vector search and conversation management components remain consistent across different LLM implementations. The fallback logic and error handling patterns are universally applicable to RAG systems.