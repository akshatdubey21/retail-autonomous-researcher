from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.retail_researcher.agent import RetailResearchAgent, ResearchRequest
from src.retail_researcher.config import Settings, SettingsError
from src.retail_researcher.knowledge_base import list_saved_reports


st.set_page_config(
    page_title="Retail Autonomous Researcher",
    page_icon="R",
    layout="wide",
)


def render_sidebar(repo_path: Path) -> None:
    st.sidebar.title("Knowledge Repository")
    saved_reports = list_saved_reports(repo_path)

    if not saved_reports:
        st.sidebar.info("No saved reports yet. Run a research task to create one.")
        return

    for report in saved_reports[:10]:
        st.sidebar.markdown(f"**{report['title']}**")
        st.sidebar.caption(report["filename"])
        st.sidebar.write(report["preview"])
        st.sidebar.divider()


def main() -> None:
    st.title("Retail Industry Autonomous Researcher")
    st.write(
        "Submit a retail-focused question and the agent will search the web, "
        "synthesize findings from multiple sources, and save the result locally."
    )

    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(settings.knowledge_repo_path)

    with st.form("research_form"):
        query = st.text_area(
            "Retail research query",
            placeholder=(
                "Example: Assess the latest AI personalization strategies used by "
                "global fashion retailers and summarize the business impact."
            ),
            height=140,
        )
        max_results = st.slider("Search results per run", min_value=3, max_value=10, value=5)
        save_report = st.checkbox("Persist report to knowledge repository", value=True)
        submitted = st.form_submit_button("Run autonomous research")

    if not submitted:
        return

    if not query.strip():
        st.warning("Enter a retail research query to continue.")
        return

    agent = RetailResearchAgent(settings)
    request = ResearchRequest(
        query=query.strip(),
        max_results=max_results,
        save_report=save_report,
    )

    with st.spinner("Research agents are gathering and synthesizing information..."):
        result = agent.run(request)

    st.subheader("Executive Summary")
    st.write(result.summary)

    st.subheader("Key Findings")
    for finding in result.key_findings:
        st.markdown(f"- {finding}")

    st.subheader("Sources")
    for source in result.sources:
        st.markdown(f"- [{source['title']}]({source['url']})")

    st.subheader("Full Report")
    st.code(result.report_text, language="markdown")

    if result.saved_path:
        st.success(f"Report saved to: {result.saved_path}")


if __name__ == "__main__":
    main()
