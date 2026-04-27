"""Chat module — RAG-style Q&A about uploaded data using FAISS + sentence-transformers."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from .ai_service import generate_chat_response


def _chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10) -> List[str]:
    """Split dataframe into text chunks for indexing."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i + chunk_size]
        text = chunk_df.to_string()
        chunks.append(text)
    return chunks


class DataChatEngine:
    """RAG engine for Q&A over uploaded data."""

    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.df = None
        self.data_context = ""
        self._initialized = False

    def is_available(self) -> bool:
        return RAG_AVAILABLE

    def initialize(self, df: pd.DataFrame, data_context: str):
        """Build FAISS index from dataframe chunks."""
        self.df = df
        self.data_context = data_context

        if not RAG_AVAILABLE:
            self._initialized = False
            return

        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.chunks = _chunk_dataframe(df)
            if not self.chunks:
                self._initialized = False
                return

            embeddings = self.model.encode(self.chunks, show_progress_bar=False)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings.astype(np.float32))
            self._initialized = True
        except Exception:
            self._initialized = False

    def query(self, question: str, chat_history: List[Dict] = None) -> str:
        """Answer a question about the data using RAG + LLM."""
        if not self._initialized:
            # Fallback: just use the data context directly
            return generate_chat_response(self.data_context, question, chat_history)

        # Retrieve top-k relevant chunks
        q_embedding = self.model.encode([question])
        D, I = self.index.search(q_embedding.astype(np.float32), k=3)
        retrieved = "\n\n".join(self.chunks[i] for i in I[0] if i < len(self.chunks))

        # Build augmented context
        augmented = f"{self.data_context}\n\nRelevant data excerpts:\n{retrieved}"
        return generate_chat_response(augmented, question, chat_history)