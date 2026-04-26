# Advanced Agentic RAG

This repository is a Vercel-native FastAPI application with a built-in dark blue web UI for an advanced bundled-corpus RAG system. The app serves the UI from `app.py`, retrieves context from `data/sample_corpus`, returns finalized cited answers from `/api/chat`, and can optionally synthesize answers with Azure OpenAI when environment variables are configured.

## Highlights

- Vercel-ready FastAPI app at `app.py`
- Dark blue browser UI served from `/`
- RAG API at `/api/chat` and `/api/query`
- Generation graph showing ingestion, chunking, indexing, retrieval, Self-RAG, generation, guardrails, and HITL
- HITL checkpoint cards for context review and final answer approval
- Source viewer with full source text and generated chunks
- Golden-set evaluation UI and `/api/evaluate`
- Corpus diagnostics at `/api/corpus` and `/api/runtime`
- BM25, semantic-overlap, hybrid, reranked, and hierarchical retrieval modes over the deployed sample corpus
- Optional Azure OpenAI answer synthesis with cited context
- No legacy local UI framework, pages, secrets, or runtime

## Deploy To Vercel

1. Push this repository to GitHub.
2. Import the repo in Vercel.
3. Keep the framework preset as FastAPI.
4. Use the default install command from `vercel.json`: `python -m pip install -r requirements.txt`.
5. Deploy.

Optional Azure OpenAI environment variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

Without those values, the app still answers with local extractive RAG over the deployed corpus.

## Endpoints

- `/` serves the Vercel UI.
- `/api` returns service metadata.
- `/api/health` returns health and corpus counts.
- `/api/runtime` returns deployment diagnostics without secret values.
- `/api/corpus` lists deployed corpus files and chunk counts.
- `/api/capabilities` lists supported retrieval, chunking, HITL, and evaluation features.
- `/api/source?path=...` returns full source content and chunks.
- `/api/chat` answers questions with finalized answer, citations, retrieved chunks, guardrails, checkpoints, and pipeline graph data.
- `/api/query` is an alias for `/api/chat`.
- `/api/evaluate` runs the bundled golden-set evaluation.

## Local Smoke Run

```cmd
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

Then open `http://127.0.0.1:8000`.
