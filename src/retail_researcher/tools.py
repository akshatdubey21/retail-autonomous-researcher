from __future__ import annotations

import os
from textwrap import shorten
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from tavily import TavilyClient


class RetailSearchInput(BaseModel):
    query: str = Field(description="Retail-focused web research query.")
    max_results: int = Field(default=5, description="Maximum number of web results to return.")


class RetailSearchTool(BaseTool):
    """Tavily-powered search tool optimized for retail market research."""

    name: str = "retail_web_search"
    description: str = (
        "Search the web for current retail industry information, strategy trends, "
        "competitive intelligence, consumer behavior, and market developments."
    )
    args_schema: Type[BaseModel] = RetailSearchInput
    _client: TavilyClient = PrivateAttr()

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._client = TavilyClient(api_key=api_key)
        os.environ["TAVILY_API_KEY"] = api_key

    def search(self, query: str, max_results: int = 5) -> dict:
        response = self._client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            topic="news",
            include_answer=True,
            include_raw_content=False,
        )

        formatted_results = []
        for item in response.get("results", []):
            formatted_results.append(
                {
                    "title": item.get("title", "Untitled source"),
                    "url": item.get("url", ""),
                    "content": shorten(item.get("content", ""), width=420, placeholder="..."),
                }
            )

        return {
            "answer": response.get("answer", ""),
            "results": formatted_results,
        }

    def run(self, query: str, max_results: int = 5) -> dict:
        return self.search(query=query, max_results=max_results)

    def _run(self, query: str, max_results: int = 5) -> str:
        result = self.search(query=query, max_results=max_results)
        lines = []
        if result.get("answer"):
            lines.append(f"Tavily answer: {result['answer']}")

        for index, item in enumerate(result.get("results", []), start=1):
            lines.append(
                f"[{index}] {item['title']}\nURL: {item['url']}\nSummary: {item['content']}"
            )

        return "\n\n".join(lines) if lines else "No search results were returned."
