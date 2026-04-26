# Architecture

The deployed application is a single Vercel FastAPI app.

- `app.py`: Vercel entrypoint, browser UI, API routes, corpus loading, retrieval, and optional Azure OpenAI synthesis
- `api/index.py`: compatibility re-export of the root FastAPI app
- `data/sample_corpus/`: bundled retrieval corpus deployed with the app
- `src/`: reusable local RAG services kept for tests and future expansion
- `tests/`: smoke tests for the RAG utilities and Vercel API surface

The Vercel path is serverless-safe: it does not depend on a local UI runtime, SQLite persistence, FAISS files, uploaded files, or local artifact state.
