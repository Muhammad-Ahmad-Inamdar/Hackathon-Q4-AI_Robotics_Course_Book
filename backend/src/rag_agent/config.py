import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# RAG settings
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))