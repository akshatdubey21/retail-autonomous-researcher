from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.retail_researcher.agent import RetailResearchAgent, ResearchRequest
from src.retail_researcher.config import Settings, SettingsError
from src.retail_researcher.knowledge_base import list_saved_reports
from src.retail_researcher.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Retail Agentic + GenAI Workspace",
    page_icon="R",
    layout="wide",
)


def render_sidebar(repo_path: Path, uploaded_files: list[str]) -> None:
    st.sidebar.title("Workspace Overview")

    st.sidebar.subheader("Indexed PDFs")
    if uploaded_files:
        for file_name in uploaded_files[:20]:
            st.sidebar.markdown(f"- {file_name}")
    else:
        st.sidebar.info("No PDF files have been uploaded yet.")

    st.sidebar.divider()
    st.sidebar.subheader("Saved Agentic Reports")
    saved_reports = list_saved_reports(repo_path)

    if not saved_reports:
        st.sidebar.info("No saved reports yet. Run a research task to create one.")
        return

    for report in saved_reports[:10]:
        st.sidebar.markdown(f"**{report['title']}**")
        st.sidebar.caption(report["filename"])
        st.sidebar.write(report["preview"])
        st.sidebar.divider()


def get_rag_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(settings)


def get_agent(settings: Settings) -> RetailResearchAgent:
    return RetailResearchAgent(settings)


def render_agentic_section(settings: Settings) -> None:
    st.subheader("Agentic AI Retail Research")
    st.write(
        "Run autonomous web research using CrewAI + Tavily + Groq and save reports in your knowledge repository."
    )

    credentials_ok = True
    try:
        settings.validate_agentic_credentials()
    except SettingsError as exc:
        credentials_ok = False
        st.warning(str(exc))

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
        submitted = st.form_submit_button("Run autonomous research", disabled=not credentials_ok)

    if not submitted:
        return

    if not query.strip():
        st.warning("Enter a retail research query to continue.")
        return

    agent = get_agent(settings)
    request = ResearchRequest(
        query=query.strip(),
        max_results=max_results,
        save_report=save_report,
    )

    with st.spinner("Research agents are gathering and synthesizing information..."):
        result = agent.run(request)

    st.markdown("### Executive Summary")
    st.write(result.summary)

    st.markdown("### Key Findings")
    for finding in result.key_findings:
        st.markdown(f"- {finding}")

    st.markdown("### Sources")
    for source in result.sources:
        st.markdown(f"- [{source['title']}]({source['url']})")

    st.markdown("### Full Report")
    st.code(result.report_text, language="markdown")

    if result.saved_path:
        st.success(f"Report saved to: {result.saved_path}")


def render_rag_section(settings: Settings, rag_pipeline: RAGPipeline) -> None:
    st.subheader("GenAI RAG Chat with Retail PDFs")
    st.write(
        "Upload one or more retail PDF files, build a local FAISS index, and ask questions grounded in document context."
    )

    uploaded_files = st.file_uploader(
        "Upload retail PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process Uploaded PDFs", type="primary"):
            with st.spinner("Extracting text, chunking, embedding, and indexing..."):
                for uploaded in uploaded_files:
                    result = rag_pipeline.ingest_pdf(uploaded.name, uploaded.getvalue())
                    st.success(
                        (
                            f"Indexed {result.file_name}: "
                            f"{result.pages_loaded} pages, {result.chunks_indexed} chunks"
                        )
                    )

    st.markdown("### Chat")
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))

    question = st.chat_input("Ask a question about your uploaded retail PDFs")
    if not question:
        return

    st.session_state.rag_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the vector index and generating answer..."):
            response = rag_pipeline.ask(question)

        source_refs = [f"{item['source']} (p.{item['page']})" for item in response.sources]
        st.write(response.answer)
        if source_refs:
            st.caption("Sources: " + ", ".join(source_refs))

        st.session_state.rag_messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": source_refs,
            }
        )


def main() -> None:
    st.title("Retail AI Command Center")
    st.write(
        "Single-page workspace with two capabilities: Agentic AI web research and GenAI RAG chat over uploaded retail PDFs."
    )

    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        st.error(str(exc))
        st.stop()

    rag_pipeline = get_rag_pipeline(settings)
    render_sidebar(settings.knowledge_repo_path, rag_pipeline.list_uploaded_files())

    tab_agentic, tab_rag = st.tabs(["Agentic AI Research", "GenAI PDF RAG"])
    with tab_agentic:
        render_agentic_section(settings)
    with tab_rag:
        render_rag_section(settings, rag_pipeline)


if __name__ == "__main__":
    main()
