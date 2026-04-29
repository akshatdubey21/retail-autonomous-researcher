
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retail_researcher.pdf_loader import DocumentChunk, load_pdf_pages, split_documents

_embedding_model_cache: dict[str, SentenceTransformer] = {}
_vector_state_cache: dict[str, dict[str, object]] = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Get or load the embedding model (module-level singleton per model name)."""
    if model_name not in _embedding_model_cache:
        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
    return _embedding_model_cache[model_name]


class RetailVectorStore:
    """A minimal FAISS-backed vector store using sentence-transformers.

    Documents are plain dicts with 'page_content' and 'metadata'.
    """

    def __init__(self, index_path: Path, embedding_model: str) -> None:
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.model = get_embedding_model(embedding_model)
        self.index_file = self.index_path / "index.faiss"
        self.meta_file = self.index_path / "index_meta.pkl"
        self._index = None
        self._metadatas: List[dict] = []
        self._texts: List[str] = []
        self._cache_key = str(self.index_path.resolve())

    def has_index(self) -> bool:
        return self.index_file.exists() and self.meta_file.exists()

    def _ensure_index(self, dim: int):
        if self._index is None:
            cached_state = _vector_state_cache.get(self._cache_key)
            if cached_state is not None:
                self._index = cached_state["index"]
                self._metadatas = cached_state["metadatas"]
                self._texts = cached_state["texts"]
                return
            if self.has_index():
                self._index = faiss.read_index(str(self.index_file))
                with open(self.meta_file, "rb") as fh:
                    payload = pickle.load(fh)
                if isinstance(payload, dict):
                    self._metadatas = payload.get("metadatas", [])
                    self._texts = payload.get("texts", [])
                else:
                    self._metadatas = list(payload)
                    self._texts = self._rebuild_texts_from_uploads(self._metadatas)
                _vector_state_cache[self._cache_key] = {
                    "index": self._index,
                    "metadatas": self._metadatas,
                    "texts": self._texts,
                }
            else:
                self._index = faiss.IndexFlatL2(dim)

    def _rebuild_texts_from_uploads(self, metadatas: list[dict]) -> list[str]:
        uploads_dir = self.index_path.parent / "uploads"
        if not uploads_dir.exists():
            return ["" for _ in metadatas]

        chunks_by_key: dict[tuple[str, int, int | None], str] = {}
        chunks_by_page: dict[tuple[str, int], list[str]] = {}

        for pdf_path in uploads_dir.glob("*.pdf"):
            pages = load_pdf_pages(pdf_path)
            chunks = split_documents(pages, chunk_size=500, chunk_overlap=50)
            for chunk in chunks:
                source = str(chunk.metadata.get("source", pdf_path.name))
                page = int(chunk.metadata.get("page", -1))
                chunk_id = chunk.metadata.get("chunk_id")
                chunks_by_key[(source, page, chunk_id)] = chunk.page_content
                chunks_by_page.setdefault((source, page), []).append(chunk.page_content)

        rebuilt: list[str] = []
        for meta in metadatas:
            source = str(meta.get("source", "unknown"))
            page = int(meta.get("page", -1))
            chunk_id = meta.get("chunk_id")
            text = chunks_by_key.get((source, page, chunk_id))
            if text is None:
                page_chunks = chunks_by_page.get((source, page), [])
                text = page_chunks[0] if page_chunks else ""
            rebuilt.append(text)
        return rebuilt

    def save(self):
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_file))
        with open(self.meta_file, "wb") as fh:
            pickle.dump({"metadatas": self._metadatas, "texts": self._texts}, fh)
        _vector_state_cache[self._cache_key] = {
            "index": self._index,
            "metadatas": self._metadatas,
            "texts": self._texts,
        }

    def add_documents(self, documents: List[DocumentChunk]) -> int:
        if not documents:
            return 0
        texts = [d.page_content for d in documents]
        vectors = np.array(self.model.encode(texts, convert_to_numpy=True), dtype="float32")
        dim = vectors.shape[1]
        self._ensure_index(dim)
        if isinstance(self._index, faiss.IndexFlatL2):
            if self._index.ntotal == 0:
                self._index.add(vectors)
            else:
                self._index.add(vectors)
        else:
            self._index.add(vectors)

        # append metadata
        for d in documents:
            self._metadatas.append(dict(d.metadata))
            self._texts.append(d.page_content)

        self.save()
        return len(documents)

    def similarity_search(self, query: str, k: int) -> List[DocumentChunk]:
        if self._index is None:
            if self.has_index():
                self._ensure_index(1)
            else:
                return []
        qvec = np.array(self.model.encode([query], convert_to_numpy=True), dtype="float32")
        D, I = self._index.search(qvec, k)
        results: List[DocumentChunk] = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self._metadatas):
                continue
            meta = self._metadatas[idx]
            text = self._texts[idx] if idx < len(self._texts) else ""
            results.append(DocumentChunk(page_content=text, metadata=meta))
        return results

    def get_all_documents(self) -> List[DocumentChunk]:
        if self._index is None and self.has_index():
            self._ensure_index(1)
        return [
            DocumentChunk(page_content=text, metadata=meta)
            for text, meta in zip(self._texts, self._metadatas)
        ]
