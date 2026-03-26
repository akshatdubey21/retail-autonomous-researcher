from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import appdirs


def _patch_crewai_storage() -> None:
    """Force CrewAI to store its local files inside the project workspace."""
    project_root = Path(__file__).resolve().parents[2]
    storage_root = project_root / ".crewai_storage"
    storage_root.mkdir(parents=True, exist_ok=True)

    def _local_user_data_dir(
        appname: str | None = None,
        appauthor: str | None = None,
        version: str | None = None,
        roaming: bool = False,
    ) -> str:
        parts = [storage_root]
        if appauthor:
            parts.append(Path(appauthor))
        if appname:
            parts.append(Path(appname))
        if version:
            parts.append(Path(version))
        return str(Path(*parts))

    appdirs.user_data_dir = _local_user_data_dir
    os.environ.setdefault("CREWAI_STORAGE_DIR", "retail_researcher")


_patch_crewai_storage()

from crewai import Agent, Crew, LLM, Task
from pydantic import BaseModel, Field

from src.retail_researcher.config import Settings
from src.retail_researcher.knowledge_base import save_report
from src.retail_researcher.tools import RetailSearchTool


class ResearchOutput(BaseModel):
    summary: str = Field(description="A concise executive summary.")
    key_findings: list[str] = Field(description="Most important retail findings.")
    opportunities: list[str] = Field(description="Business opportunities and implications.")
    risks: list[str] = Field(description="Potential risks, constraints, or caveats.")
    source_citations: list[dict[str, str]] = Field(description="Sources with title and url.")


@dataclass(slots=True)
class ResearchRequest:
    query: str
    max_results: int = 5
    save_report: bool = True


@dataclass(slots=True)
class ResearchResult:
    summary: str
    key_findings: list[str]
    sources: list[dict[str, str]]
    report_text: str
    saved_path: str | None


class RetailResearchAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.search_tool = RetailSearchTool(settings.tavily_api_key)
        self.llm = LLM(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2,
        )

    def run(self, request: ResearchRequest) -> ResearchResult:
        search_results = self.search_tool.search(request.query, max_results=request.max_results)
        search_context = self._format_search_context(search_results)

        researcher = Agent(
            role="Senior Retail Intelligence Researcher",
            goal=(
                "Investigate the user's retail-industry question using supplied search "
                "evidence and extract the strongest signals, trends, and source-backed facts."
            ),
            backstory=(
                "You are an autonomous market researcher who specializes in retail, "
                "commerce innovation, merchandising strategy, omnichannel operations, "
                "consumer behavior, and competitive intelligence."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
            tools=[self.search_tool],
        )

        analyst = Agent(
            role="Retail Strategy Analyst",
            goal=(
                "Synthesize research into an executive-quality analysis with actionable "
                "findings, business opportunities, and operational risks."
            ),
            backstory=(
                "You convert fragmented research into clear decision-ready insights for "
                "retail leaders and innovation teams."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

        research_task = Task(
            description=dedent(
                f"""
                Analyze the following retail research question:
                "{request.query}"

                First, use the retail_web_search tool to gather current evidence relevant
                to the question. Then use the provided web search material to identify:
                - dominant market trends
                - notable company examples
                - operational or commercial implications
                - data points or source-backed claims worth preserving

                Search evidence:
                {search_context}
                """
            ).strip(),
            expected_output=(
                "A structured retail research brief covering major themes, examples, "
                "and evidence-backed observations."
            ),
            agent=researcher,
        )

        synthesis_task = Task(
            description=dedent(
                """
                Convert the research brief into a JSON object with this schema:
                {
                  "summary": string,
                  "key_findings": [string],
                  "opportunities": [string],
                  "risks": [string],
                  "source_citations": [{"title": string, "url": string}]
                }

                Rules:
                - Keep the summary concise but executive-ready.
                - Base claims on the provided research brief and search evidence.
                - Include at least 3 key findings when possible.
                - Include only valid URLs in source_citations.
                """
            ).strip(),
            expected_output="A valid JSON object matching the requested schema.",
            agent=analyst,
            context=[research_task],
        )

        crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task, synthesis_task],
            verbose=False,
        )

        raw_output = crew.kickoff()
        parsed = self._parse_output(str(raw_output), search_results)
        report_text = self._build_report_text(request.query, parsed)

        saved_path = None
        if request.save_report:
            saved_path = str(save_report(self.settings.knowledge_repo_path, request.query, report_text))

        return ResearchResult(
            summary=parsed.summary,
            key_findings=parsed.key_findings,
            sources=parsed.source_citations,
            report_text=report_text,
            saved_path=saved_path,
        )

    def _format_search_context(self, search_results: dict) -> str:
        lines = []
        if search_results.get("answer"):
            lines.append(f"Tavily answer: {search_results['answer']}")

        for index, item in enumerate(search_results.get("results", []), start=1):
            lines.append(
                f"[{index}] {item['title']}\nURL: {item['url']}\nSummary: {item['content']}"
            )
        return "\n\n".join(lines)

    def _parse_output(self, raw_output: str, search_results: dict) -> ResearchOutput:
        try:
            candidate = raw_output.strip()
            if "```json" in candidate:
                candidate = candidate.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0]
            elif candidate.startswith("```"):
                candidate = candidate.split("```", maxsplit=1)[1].rsplit("```", maxsplit=1)[0]

            data = json.loads(candidate.strip())
            if hasattr(ResearchOutput, "model_validate"):
                return ResearchOutput.model_validate(data)
            return ResearchOutput.parse_obj(data)
        except Exception:
            fallback_sources = [
                {"title": item["title"], "url": item["url"]}
                for item in search_results.get("results", [])
                if item.get("url")
            ]
            return ResearchOutput(
                summary=raw_output.strip()[:800] or "Research completed.",
                key_findings=[
                    "The agent produced a narrative response that could not be parsed into structured JSON.",
                    "Review the saved report for the full synthesized output.",
                ],
                opportunities=[],
                risks=[],
                source_citations=fallback_sources,
            )

    def _build_report_text(self, query: str, output: ResearchOutput) -> str:
        def format_lines(items: list[str]) -> str:
            if not items:
                return "- None identified."
            return "\n".join(f"- {item}" for item in items)

        sources = output.source_citations or []
        source_block = (
            "\n".join(f"- {item['title']}: {item['url']}" for item in sources)
            if sources
            else "- No sources captured."
        )

        return dedent(
            f"""
            # Retail Research Report

            Query: {query}

            Executive Summary:
            {output.summary}

            Key Findings:
            {format_lines(output.key_findings)}

            Opportunities:
            {format_lines(output.opportunities)}

            Risks:
            {format_lines(output.risks)}

            Sources:
            {source_block}
            """
        ).strip()
