# Retail Agentic + GenAI Workspace

A production-ready single-page retail AI workspace built with Python and Streamlit.

The application combines:
- Agentic AI web research (CrewAI + Tavily + Groq)
- GenAI RAG chat over uploaded retail PDFs (PyPDF + LangChain chunking + HuggingFace embeddings + FAISS + local FLAN-T5)

## Features

- Single UI page with two integrated tabs:
  - Agentic AI retail web research
  - GenAI RAG chat with uploaded PDFs
- Local vector index persistence using FAISS
- Source references in RAG answers (filename + page)
- Text-based knowledge repository for saved agentic reports

## Project Structure

```text
.
├── app.py
├── data/
├── knowledge_repo/
├── requirements.txt
└── src/
    └── retail_researcher/
        ├── agent.py
        ├── config.py
        ├── knowledge_base.py
    ├── llm.py
    ├── pdf_loader.py
    ├── rag_pipeline.py
        └── tools.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and populate values:

```env
# Required for Agentic AI tab
GROQ_API_KEY=...
TAVILY_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile

# Used by GenAI RAG tab (all local, free)
HF_MODEL_NAME=google/flan-t5-base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
FAISS_INDEX_PATH=data/faiss_index
UPLOAD_DIR=data/uploads
MAX_NEW_TOKENS=512
RAG_TOP_K=4
```

4. Start the app:

```bash
streamlit run app.py
```

## Docker

Build the image:

```bash
docker build -t retail-autonomous-researcher .
```

Run it from Docker Desktop or the CLI:

```bash
docker run --rm -p 8501:8501 --env-file .env -v "${PWD}\\knowledge_repo:/app/knowledge_repo" retail-autonomous-researcher
```

Open `http://localhost:8501` after the container starts.

If Docker Desktop is not already running on Windows, start it first so the Linux engine is available.

## How It Works

### Agentic AI mode

1. The user submits a retail research prompt.
2. Tavily search is used to gather high-quality web results.
3. A CrewAI workflow uses specialized agents to research and synthesize findings.
4. The final report is rendered in Streamlit and saved as a text document under `knowledge_repo/`.

### GenAI RAG mode

1. User uploads one or more retail PDFs.
2. Text is extracted per page using PyPDF.
3. Text is chunked with RecursiveCharacterTextSplitter.
4. Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
5. Embeddings are stored in a local FAISS index.
6. User asks a question.
7. Top-k chunks are retrieved from FAISS.
8. `google/flan-t5-base` generates a grounded answer from retrieved context.
9. App returns answer with source references.

## Example Queries

- Analyze the latest omnichannel trends shaping grocery retail in India.
- Research how AI-driven pricing is affecting apparel retail margins.
- Summarize current strategies top retailers use to reduce cart abandonment.

## Notes

- Agentic AI features require valid Groq and Tavily API keys.
- RAG features run locally and do not require paid API keys.
- Reports are saved locally as plain text for easy reuse and versioning.
