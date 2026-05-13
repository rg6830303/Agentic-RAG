# Advanced Agentic RAG

This repository is a Vercel-native FastAPI application with a built-in dark blue AI chat UI for an advanced Agentic RAG system. The app serves the UI from `app.py`, retrieves context from `data/sample_corpus`, can augment answers with text-only Wikipedia context, returns finalized cited answers from `/api/chat`, and can optionally synthesize answers with Azure OpenAI when environment variables are configured.

## Highlights

- Vercel-ready FastAPI app at `app.py`
- Stitch-inspired production dark blue browser chat UI served from `/`
- Enter-to-send prompt composer; use Shift+Enter for multiline prompts
- Multi-turn saved chat threads with new chat, resume, clone, edit prompt, regenerate, agenda, suggestions, and local browser fallback
- RAG API at `/api/chat` and `/api/query`
- Generation graph showing ingestion, chunking, indexing, retrieval, Self-RAG, generation, guardrails, and HITL
- HITL checkpoint cards for context review and final answer approval
- Source viewer with full source text and generated chunks
- Golden-set evaluation UI and `/api/evaluate`
- Corpus diagnostics at `/api/corpus` and `/api/runtime`
- BM25, semantic-overlap, hybrid, reranked, and hierarchical retrieval modes over the deployed sample corpus
- Lightweight Wikipedia text retrieval through the official MediaWiki API with URL-backed citations
- Optional Azure OpenAI answer synthesis with cited context
- No legacy local UI framework, pages, secrets, or runtime

## Chat UX

- Press `Enter` in the composer to send a non-empty prompt.
- Press `Shift+Enter` to insert a new line.
- The Send button uses the same submit path as the keyboard shortcut.
- Duplicate sends are blocked while retrieval/generation is running.
- Loading feedback shows a retrieval/generation state while the RAG pipeline runs.
- Empty state prompt chips help start common workflows such as summarizing the corpus, comparing retrieved sources, or asking for Wikipedia-backed context.
- A single chat thread can contain many user prompts and assistant answers. Sending a follow-up appends to the active thread; it does not require creating a new chat.
- User prompts have an Edit action. Saving an edit creates a clean branch by removing later turns and regenerating the assistant answer for the edited prompt.
- The latest assistant answer has a Regenerate action that replaces that answer while preserving the thread structure.

## Mobile UX

- Phone and tablet layouts use a compact sticky header with the current chat title and status.
- The desktop sidebar becomes a slide-out drawer on smaller screens, with New Chat, saved chats, Knowledge Base, and Diagnostics still available.
- Retrieval settings and advanced answer details collapse on mobile so the chat remains the primary surface.
- Citations, source URLs, source paths, diagnostics rows, and retrieved chunks wrap or collapse to avoid page-level horizontal scrolling.

## Stitch UI Integration

The current UI is integrated from the Google Stitch export in `design/stitch-export/stitch_agentic_rag_enterprise_interface/`. The FastAPI app still serves a single HTML/CSS/JS shell from `app.py` for Vercel compatibility, but the visual system, navigation, chat layout, knowledge base cards, source viewer treatment, and diagnostics styling follow the Stitch screens and `DESIGN.md` tokens.

Stitch screens used:

- `agentic_rag_active_chat_2`
- `agentic_rag_new_chat`
- `agentic_rag_knowledge_base_2`
- `agentic_rag_diagnostics_2`
- `agentic_rag_source_viewer`
- `agentic_rag_mobile_chat_2`

## Saved Chats

Chat sessions are saved automatically after each answer. Each session stores:

- `session_id`
- title/session name
- created and updated timestamps
- ordered prompt/answer history plus normalized message IDs for user and assistant messages
- citations and source snippets when available
- retrieval metadata such as provider, retrieval mode, confidence, and source counts
- a thread-specific agenda and compact memory summary

The UI loads saved sessions in the left sidebar and can resume or clone an older chat. Opening an old thread restores the full transcript and follow-up prompts append to that same thread. Thread memory is built from the agenda, compact older-summary, and recent turns, and is sent only for that active thread so separate chats do not mix context.

The browser keeps the active thread transcript as the UI source of truth, appends new turns locally, and sends the active `session_id` plus compact thread memory to the backend for each follow-up. The backend stores history under `data/chat_history/` for local runs. On Vercel/serverless deployments, filesystem writes may be ephemeral, so the browser UI also keeps a best-effort `localStorage` cache for that browser. This is not a multi-user durable database.

## Wikipedia Text Retrieval

Wikipedia support is text-only and enabled by default. The app uses the official MediaWiki API to search for relevant pages and fetch plaintext extracts. Wikipedia chunks are merged with the existing local corpus retrieval results and are cited separately from local corpus sources.

Wikipedia citation fields include:

- `source_type: "wikipedia"`
- page title
- source URL
- retrieved snippet or extract text

If Wikipedia lookup fails, times out, or returns no relevant result, the app gracefully falls back to the bundled corpus. No API key or new environment variable is required. Set `AGENTIC_RAG_DISABLE_WIKIPEDIA=1` only if you want to disable Wikipedia retrieval for a local or hosted environment.

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
- `/api/chat/history` lists saved chat sessions.
- `/api/chat/history/{session_id}` returns a saved chat transcript.
- `/api/chat/history/{session_id}/clone` clones an existing saved chat.
- `/api/chats` provides compatibility aliases for listing and creating sessions.
- `/api/chats/{session_id}` provides compatibility aliases for retrieving, updating, or deleting sessions.
- `/api/chats/{session_id}/messages` appends a prompt to an existing thread and returns the generated answer.
- `/api/chats/{session_id}/messages/{message_id}` edits a user prompt, branches from that turn, and regenerates the answer.
- `/api/chats/{session_id}/regenerate` regenerates the latest assistant response in a thread.
- `/api/evaluate` runs the bundled golden-set evaluation.

## Local Smoke Run

```cmd
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

Then open `http://127.0.0.1:8000`.
