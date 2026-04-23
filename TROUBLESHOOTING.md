# Troubleshooting

## `AGENTS.md` Missing

This repository currently does not include `AGENTS.md`. The implementation therefore follows the explicit user request and the repository-local structure created for this project.

## FAISS Unavailable

If FAISS rebuilds fail, install `faiss-cpu` in the active Python environment. BM25 retrieval still works independently.

## PDF or DOCX Ingestion Fails

Install the optional loaders:

- `pypdf`
- `python-docx`

## Azure Chat or Embeddings Missing

- Missing chat deployment: the app falls back to local extractive answer generation
- Missing embeddings deployment: FAISS rebuild and vector retrieval remain unavailable, but BM25 continues to work

## Streamlit Starts but Pages Look Empty

Ingest sample files first from `data/sample_corpus/ncert_physics/`, then rebuild indexes if needed.
