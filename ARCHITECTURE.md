# Architecture

## Layout

- `app.py`: thin Streamlit bootstrap that initializes settings, cached services, and landing navigation
- `pages/`: one page per feature area
- `src/config/`: environment loading, local path setup, and diagnostics
- `src/providers/`: Azure OpenAI abstraction
- `src/ingestion/`: file loading and ingestion workflow
- `src/chunking/`: chunk strategy selection and chunk generation
- `src/docstore/`: SQLite-backed metadata and chunk persistence
- `src/indexing/`: FAISS and BM25 local indexes with rebuild orchestration
- `src/retrieval/`: hybrid retrieval orchestration with hierarchical expansion
- `src/reranking/`: lightweight reranking
- `src/agentic/`: Self-RAG answer generation, reflection, and guardrails
- `src/evaluation/`: golden dataset loading, heuristic evaluation, and optional RAGAS adapter
- `src/ui/`: Streamlit shared helpers, state initialization, and renderers

## Data Flow

1. Files are loaded locally and normalized into `LoadedDocument` records.
2. Chunking chooses a manual or auto-selected strategy per file.
3. Approved chunks are persisted in SQLite and exported to local JSONL artifacts.
4. BM25 is rebuilt directly from local chunk text. FAISS is rebuilt from Azure embedding vectors when configured.
5. Chat queries fan out into vector and sparse branches in parallel when enabled.
6. Retrieved contexts can be reviewed manually, reranked, expanded hierarchically, and passed through Self-RAG reflection.
7. Final answers, citations, checkpoints, and evaluation artifacts stay local on disk.
