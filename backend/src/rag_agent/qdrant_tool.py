from qdrant_client import QdrantClient, models
from qdrant_client.models import PayloadSchemaType
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from .config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME


class QdrantRAGTool:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # 🔒 Ensure payload index exists (ONE-TIME, SAFE)
        self._ensure_chapter_index()

    def _ensure_chapter_index(self):
        """
        Ensure Qdrant payload index exists for chapter filtering.
        This prevents 400 Bad Request errors.
        """
        try:
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="chapter",
                field_schema=PayloadSchemaType.INTEGER
            )
            print("✅ Qdrant payload index ensured for 'chapter'")
        except Exception as e:
            # Index already exists → safe to ignore
            print("ℹ️ Qdrant index check:", str(e))

    def retrieve_context(self, query: str, chapter: Optional[int] = None) -> List[str]:
        vector = self.embedder.encode(query).tolist()

        # 🎯 Optional chapter filter
        query_filter = None
        if chapter is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chapter",
                        match=models.MatchValue(value=chapter)
                    )
                ]
            )

        # 🔍 First attempt (with filter if available)
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            query_filter=query_filter,
            limit=10,
            with_payload=True
        )

        # 🔁 Fallback: remove filter if nothing found
        if not results and chapter is not None:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=10,
                with_payload=True
            )

        contexts: List[str] = []
        for hit in results:
            payload = hit.payload or {}
            text = payload.get("text", "")
            chapter_val = payload.get("chapter", "N/A")
            page_val = payload.get("page", "N/A")

            meta = f"(Chapter: {chapter_val}, Page: {page_val})"
            contexts.append(f"{meta} {text}")

        return contexts
