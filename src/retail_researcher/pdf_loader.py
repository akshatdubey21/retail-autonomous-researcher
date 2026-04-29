from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass(slots=True)
class DocumentChunk:
    page_content: str
    metadata: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def load_pdf_pages(pdf_path: Path) -> List[DocumentChunk]:
    """Return a list of page chunks with attribute and dict-style access."""
    reader = PdfReader(str(pdf_path))
    documents: list[DocumentChunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(DocumentChunk(page_content=text, metadata={"source": pdf_path.name, "page": page_number}))

    return documents


def split_documents(documents: List[DocumentChunk], chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
    """Simple character-based splitter that preserves metadata."""
    chunks: list[DocumentChunk] = []
    for doc in documents:
        text = doc.page_content
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(page_content=chunk_text, metadata=dict(doc.metadata)))
            if end == length:
                break
            start = max(0, end - chunk_overlap)

    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = idx
    return chunks
