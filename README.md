# Advanced Agentic RAG

This repository is a Vercel-native FastAPI application with a built-in web UI for a bundled-corpus RAG system. The app serves the UI from `app.py`, retrieves context from `data/sample_corpus`, returns cited answers from `/api/chat`, and can optionally synthesize answers with Azure OpenAI when Vercel environment variables are configured.

## Highlights

- Vercel-ready FastAPI app at `app.py`
- Browser UI served from `/`
- RAG API at `/api/chat` and `/api/query`
- Corpus diagnostics at `/api/corpus` and `/api/runtime`
- Local BM25-style retrieval over the deployed sample corpus
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
- `/api/chat` answers questions with retrieved context and citations.
- `/api/query` is an alias for `/api/chat`.

## Local Smoke Run

```cmd
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

Then open `http://127.0.0.1:8000`.
