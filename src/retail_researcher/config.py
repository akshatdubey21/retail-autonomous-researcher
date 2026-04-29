from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class SettingsError(RuntimeError):
    """Raised when required application settings are missing."""


@dataclass(slots=True)
class Settings:
    groq_api_key: str
    tavily_api_key: str
    groq_model: str
    knowledge_repo_path: Path
    hf_model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    faiss_index_path: Path
    upload_dir: Path
    max_new_tokens: int
    rag_top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        knowledge_repo_path = Path("knowledge_repo").resolve()
        hf_model_name = os.getenv("HF_MODEL_NAME", "google/flan-t5-base").strip()
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
        chunk_size = _get_int_env("CHUNK_SIZE", 500)
        chunk_overlap = _get_int_env("CHUNK_OVERLAP", 50)
        faiss_index_path = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index")).resolve()
        upload_dir = Path(os.getenv("UPLOAD_DIR", "data/uploads")).resolve()
        max_new_tokens = _get_int_env("MAX_NEW_TOKENS", 512)
        rag_top_k = _get_int_env("RAG_TOP_K", 4)

        knowledge_repo_path.mkdir(parents=True, exist_ok=True)
        faiss_index_path.mkdir(parents=True, exist_ok=True)
        upload_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            groq_api_key=groq_api_key,
            tavily_api_key=tavily_api_key,
            groq_model=groq_model,
            knowledge_repo_path=knowledge_repo_path,
            hf_model_name=hf_model_name,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            faiss_index_path=faiss_index_path,
            upload_dir=upload_dir,
            max_new_tokens=max_new_tokens,
            rag_top_k=rag_top_k,
        )

    def validate_agentic_credentials(self) -> None:
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            joined = ", ".join(missing)
            raise SettingsError(
                f"Missing required environment variable(s): {joined}. "
                "Agentic web research is unavailable until these are set."
            )


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"Environment variable {name} must be an integer.") from exc
    if value <= 0:
        raise SettingsError(f"Environment variable {name} must be greater than 0.")
    return value
