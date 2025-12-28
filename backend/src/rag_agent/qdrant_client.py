from qdrant_client import QdrantClient
from .config import QDRANT_URL, QDRANT_API_KEY


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

    def search(self, collection_name, query_vector, limit=5):
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
