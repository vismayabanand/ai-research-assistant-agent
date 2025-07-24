# src/rag_qa.py
"""RAG utilities that use ChromaDB + Gemini 1.5 embeddings/chat."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from chromadb import PersistentClient, EmbeddingFunction
import google.generativeai as genai

# ... (Environment and GeminiEmbeddingFunction class remain the same) ...
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
API_KEY = os.environ["GOOGLE_API_KEY"]
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
genai.configure(api_key=API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    MODEL = "models/embedding-001"
    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in input:
            resp = genai.embed_content(
                model=self.MODEL,
                content=text,
                task_type="retrieval_document",
            )
            vectors.append(resp["embedding"])
        return vectors

_embedder = GeminiEmbeddingFunction()
client = PersistentClient(path=CHROMA_DIR)
MODEL_NAME = "gemini-1.5-flash-latest"

# **MODIFIED FUNCTION**
def build_rag(documents: List[str], metadatas: List[Dict[str, Any]], *, collection_name: str = "papers"):
    """Create (or reset) a Chroma collection populated with text chunks."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.create_collection(collection_name, embedding_function=_embedder)

    # The function now directly uses the provided documents (chunks) and metadatas
    ids = [str(i) for i in range(len(documents))]
    col.add(ids=ids, documents=documents, metadatas=metadatas)
    return col


def answer_query(collection, query: str, *, k: int = 5) -> str: # Increased k for better context
    """Retrieve top‑k docs, then ask Gemini to answer using that context."""
    res = collection.query(query_texts=[query], n_results=k)

    snippets = []
    # Add a check for documents to avoid errors if nothing is returned
    if res["documents"]:
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            snippets.append(f"Source: {meta['title']}\nAuthors: {meta.get('authors', 'N/A')}\nContent: {doc}")
    
    if not snippets:
        return "I couldn't find any relevant information in the provided papers."

    context_block = "\n\n".join(snippets)

    prompt = (
        "You are an academic assistant. Using ONLY the context below, "
        "answer the question. If the answer isn't in context, say 'I don't know.'\n\n"
        f"--- CONTEXT ---\n{context_block}\n\n--- QUESTION ---\n{query}\n\nAnswer:"
    )

    answer = genai.GenerativeModel(MODEL_NAME).generate_content(prompt).text.strip()
    return answer