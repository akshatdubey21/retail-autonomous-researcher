# Retail Autonomous Researcher

An advanced retail-industry research assistant built with Python, Streamlit, LangChain, and CrewAI. The app uses a Groq-hosted LLM plus Tavily web search to investigate a user question, synthesize multi-source findings, and save a structured text report into a local knowledge repository.

## Features

- Retail-focused autonomous researcher workflow
- Groq-backed LLM for fast synthesis
- Tavily web search for current web discovery
- CrewAI orchestration with explicit researcher and analyst roles
- Streamlit interface for running and reviewing research
- Text-based knowledge repository for persistent reports

## Project Structure

```text
.
├── app.py
├── knowledge_repo/
├── requirements.txt
└── src/
    └── retail_researcher/
        ├── agent.py
        ├── config.py
        ├── knowledge_base.py
        └── tools.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and populate:

```env
GROQ_API_KEY=...
TAVILY_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

4. Start the app:

```bash
streamlit run app.py
```

## How It Works

1. The user submits a retail research prompt.
2. Tavily search is used to gather high-quality web results.
3. A CrewAI workflow uses specialized agents to research and synthesize findings.
4. The final report is rendered in Streamlit and saved as a text document under `knowledge_repo/`.

## Example Queries

- Analyze the latest omnichannel trends shaping grocery retail in India.
- Research how AI-driven pricing is affecting apparel retail margins.
- Summarize current strategies top retailers use to reduce cart abandonment.

## Notes

- The app expects valid Groq and Tavily API keys.
- Reports are saved locally as plain text for easy reuse and versioning.
