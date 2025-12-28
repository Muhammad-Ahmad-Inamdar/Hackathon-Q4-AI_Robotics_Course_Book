# RAG Chatbot Backend

This is a Retrieval-Augmented Generation (RAG) chatbot backend built with FastAPI, Qdrant, and Google Gemini.

## Features

- RAG-based question answering using vector search
- Chapter filtering for targeted queries
- Integration with Google Gemini for intelligent responses
- Qdrant for vector storage and similarity search
- Sentence transformers for text embeddings

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file based on `.env.example` and add your API keys and configuration.

3. Run the server:
   ```bash
   python main.py
   ```

## Endpoints

- `GET /` - Health check
- `POST /chat` - Main chat endpoint with optional chapter filtering
- `POST /simple_chat` - Simple chat endpoint without chapter filtering

## Environment Variables

- `QDRANT_URL` - URL for Qdrant vector database
- `QDRANT_API_KEY` - API key for Qdrant
- `COLLECTION_NAME` - Name of the Qdrant collection
- `GEMINI_API_KEY` - Google Gemini API key
- `TOP_K` - Number of top results to retrieve (default: 4)
- `SIMILARITY_THRESHOLD` - Minimum similarity threshold (default: 0.25)