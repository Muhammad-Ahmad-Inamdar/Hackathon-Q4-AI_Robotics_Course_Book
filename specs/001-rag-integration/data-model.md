# Data Model: RAG System Integration & Cleanup

## Entities

### Query
**Description**: A text-based question submitted by the user
**Fields**:
- id: String (unique identifier)
- content: String (the actual query text from user)
- timestamp: DateTime (when the query was submitted)
- userId: String (optional, for tracking purposes)
- status: Enum ['pending', 'processing', 'completed', 'failed'] (current processing state)

### Response
**Description**: The AI-generated answer to the user's query
**Fields**:
- id: String (unique identifier, matches associated query)
- queryId: String (reference to the original query)
- content: String (the AI-generated response text)
- sources: Array of Objects (references to textbook fragments used)
- timestamp: DateTime (when response was generated)
- confidence: Number (0-100, confidence score of the response)

### Textbook Content
**Description**: The source material from "Physical AI & Humanoid Robotics" textbook
**Fields**:
- id: String (unique identifier for content fragment)
- title: String (chapter/section title)
- content: String (the actual text content)
- pageNumber: Number (original page number in textbook)
- chapterNumber: Number (chapter in the textbook)
- sectionNumber: String (section identifier)
- embedding: Array of Numbers (vector representation for similarity search)

## Relationships
- Query → Response (one-to-one): Each query generates one response
- Response → Textbook Content (one-to-many): Each response may reference multiple textbook fragments

## Validation Rules
- Query content must be non-empty and under 1000 characters
- Response must be generated within 10 seconds or marked as timeout
- Textbook content fragments must be properly segmented and indexed

## State Transitions
- Query status: pending → processing → [completed | failed]
- Response is created after successful query processing