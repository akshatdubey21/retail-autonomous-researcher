from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.retail_researcher.config import Settings
from src.retail_researcher.llm import LocalFlanGenerator
from src.retail_researcher.pdf_loader import load_pdf_pages, split_documents
from src.retail_researcher.vector_store import RetailVectorStore

_query_cache: dict[str, str] = {}


@dataclass(slots=True)
class IngestResult:
    file_name: str
    pages_loaded: int
    chunks_indexed: int


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    sources: list[dict[str, str | int]]


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_store = RetailVectorStore(settings.faiss_index_path, settings.embedding_model)
        self.generator = LocalFlanGenerator(settings.hf_model_name, settings.max_new_tokens)

    def ingest_pdf(self, file_name: str, file_bytes: bytes) -> IngestResult:
        target_path = self.settings.upload_dir / file_name
        target_path.write_bytes(file_bytes)

        pages = load_pdf_pages(target_path)
        chunks = split_documents(
            documents=pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        # vector_store expects plain dict documents
        chunks_indexed = self.vector_store.add_documents(chunks)

        return IngestResult(
            file_name=file_name,
            pages_loaded=len(pages),
            chunks_indexed=chunks_indexed,
        )

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        # Check cache first
        question_lower = question.lower().strip()
        question_hash = hashlib.md5(question_lower.encode()).hexdigest()
        
        if question_hash in _query_cache:
            # Reconstruct cached RAGAnswer from stored JSON string
            import json
            cached_data = json.loads(_query_cache[question_hash])
            return RAGAnswer(
                answer=cached_data["answer"],
                sources=cached_data["sources"],
            )
        
        is_summary_request = any(keyword in question_lower for keyword in ["summary", "summarize", "summarise", "overview"])

        if is_summary_request:
            docs = self.vector_store.get_all_documents()
        else:
            k = top_k or self.settings.rag_top_k
            docs = self.vector_store.similarity_search(question, k=k)
        if not docs:
            result = RAGAnswer(
                answer="No indexed documents found. Upload and process at least one PDF first.",
                sources=[],
            )
        else:
            context = "\n\n".join((self._get_page_content(doc) or "") for doc in docs)
            if is_summary_request:
                summary_question = (
                    "Summarize this document in plain language in 4-6 concise bullet points. "
                    "Focus on the main facts, findings, and important values. Do not repeat headings or boilerplate."
                )
                answer = self.generator.answer(question=summary_question, context=context)
            else:
                answer = self.generator.answer(question=question, context=context)

            seen = set()
            sources: list[dict[str, str | int]] = []
            for doc in docs:
                metadata = self._get_metadata(doc)
                source = str(metadata.get("source", "unknown"))
                page = int(metadata.get("page", -1))
                key = (source, page)
                if key in seen:
                    continue
                seen.add(key)
                sources.append({"source": source, "page": page})

            result = RAGAnswer(answer=answer, sources=sources)
        
        # Cache the result
        import json
        _query_cache[question_hash] = json.dumps({"answer": result.answer, "sources": result.sources})
        
        return result

    def _get_metadata(self, doc: object) -> dict:
        if isinstance(doc, dict):
            return dict(doc.get("metadata", {}))
        return dict(getattr(doc, "metadata", {}) or {})

    def _get_page_content(self, doc: object) -> str:
        if isinstance(doc, dict):
            return str(doc.get("page_content", ""))
        return str(getattr(doc, "page_content", ""))

    def list_uploaded_files(self) -> list[str]:
        return sorted([path.name for path in self.settings.upload_dir.glob("*.pdf")], reverse=True)
