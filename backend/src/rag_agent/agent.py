from google import genai
from typing import Optional
from .qdrant_tool import QdrantRAGTool
from .config import GEMINI_API_KEY


class RAGAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.qdrant_tool = QdrantRAGTool()
        self.model_id = "gemini-2.0-flash"

    def infer_chapter(self, query: str) -> Optional[int]:
        q = query.lower()

        if "pid" in q or "controller" in q:
            return 4
        if "robotics" in q or "robot" in q:
            return 1
        if "ros" in q:
            return 6

        return None

    def ask_gemini(self, context: str, question: str) -> str:
        prompt = f"""
You are a reasoning engine.
Answer ONLY from the provided context.

CONTEXT:
{context}

QUESTION:
{question}
"""
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config={"temperature": 0.2}
        )
        return response.text

    def process_query(self, query: str, chapter: Optional[int] = None) -> dict:
        if chapter is None:
            chapter = self.infer_chapter(query)

        contexts = self.qdrant_tool.retrieve_context(query, chapter)

        if not contexts:
            return {
                "answer": "This question is outside the scope of this book.",
                "sources_used": 0,
                "filter_applied": "None",
                "context_depth": "Low"
            }

        combined_context = "\n\n".join(contexts)
        answer = self.ask_gemini(combined_context, query)

        return {
            "answer": answer,
            "sources_used": len(contexts),
            "filter_applied": f"Chapter {chapter}" if chapter else "Auto",
            "context_depth": "High"
        }
