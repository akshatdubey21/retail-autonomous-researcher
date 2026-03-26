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

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        knowledge_repo_path = Path("knowledge_repo").resolve()

        missing = []
        if not groq_api_key:
            missing.append("GROQ_API_KEY")
        if not tavily_api_key:
            missing.append("TAVILY_API_KEY")

        if missing:
            joined = ", ".join(missing)
            raise SettingsError(
                f"Missing required environment variable(s): {joined}. "
                "Create a .env file based on .env.example."
            )

        knowledge_repo_path.mkdir(parents=True, exist_ok=True)
        return cls(
            groq_api_key=groq_api_key,
            tavily_api_key=tavily_api_key,
            groq_model=groq_model,
            knowledge_repo_path=knowledge_repo_path,
        )
