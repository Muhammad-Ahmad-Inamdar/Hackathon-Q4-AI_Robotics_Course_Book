# Quickstart: RAG System Integration & Cleanup

## Prerequisites

- Python 3.11+ installed
- Node.js 18+ installed
- Git installed
- Google AI API key for Gemini 2.0 Flash-Lite

## Setup

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Google AI API key
```

### 3. Frontend Setup
```bash
cd ../frontend  # or root directory if Docusaurus is in root
npm install
```

### 4. Environment Configuration
Create/update `.env` file in backend:
```
GOOGLE_API_KEY=your_google_ai_api_key_here
QDRANT_URL=http://localhost:6333
```

## Running the System

### 1. Start Qdrant Vector Database
```bash
# Option A: Using Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Option B: If you have local installation
# Follow Qdrant installation guide
```

### 2. Start Backend API
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

### 3. Start Frontend
```bash
cd frontend  # or root if Docusaurus is in root
npm start
```

## Ingest Textbook Content

Before using the chatbot, you need to ingest the textbook content:

```bash
cd backend
python -c "
from src.rag_agent.qdrant_tool import ingest_textbook_content
ingest_textbook_content('path/to/your/textbook/files')
"
```

## Using the Chatbot

1. Open your browser to `http://localhost:3000`
2. Use the chat interface to ask questions about the textbook
3. Try asking "How many chapters are in this book?" to test the reasoning engine

## API Endpoints

### POST /chat
- Request: `{"query": "your question here"}`
- Response: `{"response": "AI answer", "sources": [...], "confidence": 95}`

### POST /ingest
- Request: `{"textbook_path": "/path/to/textbook"}`
- Response: `{"status": "success", "chunks_processed": 120}`

## Cleanup Process

To run the cleanup process:
```bash
# Remove temporary files, logs, and duplicates
find . -name "*.tmp" -o -name "*.log" -o -name "*.temp" | xargs rm -f
```

## Troubleshooting

### Common Issues

**Issue**: Qdrant connection error
**Solution**: Ensure Qdrant is running on http://localhost:6333

**Issue**: API key not found
**Solution**: Verify GOOGLE_API_KEY is set in your .env file

**Issue**: Slow response times
**Solution**: Check that textbook content has been properly ingested into the vector database