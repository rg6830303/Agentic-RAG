# Advanced Agentic RAG

This repository is a Vercel-native FastAPI application with a built-in dark blue AI chat UI for an advanced Agentic RAG system. The app serves the UI from `app.py`, retrieves context from `data/sample_corpus`, can augment answers with text-only Wikipedia context, returns finalized cited answers from `/api/chat`, and can optionally synthesize answers with Azure OpenAI when environment variables are configured.

## Highlights

- Vercel-ready FastAPI app at `app.py`
- Stitch-inspired production dark blue browser chat UI served from `/`
- Login/sign up with signed HttpOnly session cookies and password hashing
- Per-user chat threads stored in the backend, isolated by authenticated account
- Enter-to-send prompt composer; use Shift+Enter for multiline prompts
- Multi-turn saved chat threads with new chat, resume, clone, edit prompt, regenerate, agenda, and suggestions
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
- While editing a user prompt, `Enter` saves and regenerates, `Shift+Enter` inserts a newline, and `Escape` cancels the edit.
- Loading feedback shows a retrieval/generation state while the RAG pipeline runs.
- Empty state prompt chips help start common workflows such as summarizing the corpus, comparing retrieved sources, or asking for Wikipedia-backed context.
- A single chat thread can contain many user prompts and assistant answers. Sending a follow-up appends to the active thread; it does not require creating a new chat.
- User prompts have an Edit action. Saving an edit creates a clean branch by removing later turns and regenerating the assistant answer for the edited prompt.
- The latest assistant answer has a Regenerate action that replaces that answer while preserving the thread structure.
- Assistant messages keep the main chat clean: citations, source chunks, guardrails, confidence, and pipeline metrics open from the per-message Details/Sources drawer.

## Authentication

The production UI is account-first. Logged-out users see a login/sign-up screen, and logged-in users see only their own chat threads.

- `POST /api/auth/signup` creates an account and sets a signed HttpOnly cookie.
- `POST /api/auth/login` verifies the password and creates the session.
- `POST /api/auth/logout` clears the session cookie.
- `GET /api/auth/me` returns the current authenticated user.

Passwords are hashed with PBKDF2-HMAC-SHA256 and per-user salts. The frontend never receives password hashes or session secrets, and chat-thread endpoints resolve the user from the signed cookie instead of trusting a frontend `user_id`.

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

Authenticated chat sessions are saved automatically after each answer. Each session stores:

- `session_id`
- title/session name
- created and updated timestamps
- ordered prompt/answer history plus normalized message IDs for user and assistant messages
- citations and source snippets when available
- retrieval metadata such as provider, retrieval mode, confidence, and source counts
- a thread-specific agenda and compact memory summary

The UI loads only the signed-in user's sessions in the left sidebar and can resume or clone an older chat. Opening an old thread restores the full transcript and follow-up prompts append to that same thread. Thread memory is built from the agenda, compact older-summary, and recent turns, and is sent only for that active thread so separate chats and separate users do not mix context.

For authenticated users, the backend SQL store is the source of truth. If `DATABASE_URL` points to Postgres, the app uses it for accounts, threads, and messages. Without `DATABASE_URL`, local development falls back to SQLite at `data/agentic_rag_app.sqlite3`. On Vercel without `DATABASE_URL`, the app can run signup/login against temporary `/tmp` SQLite storage so the UI remains usable, but those accounts and chats are not durable and may reset after cold starts. Legacy unauthenticated `/api/chat` calls still use the older file-based history under `data/chat_history/` for backwards compatibility.

On Vercel, the app does not initialize the account database during import. If the configured database is unreachable, `/` still loads and `/api/health` plus `/api/runtime` still return diagnostic JSON. Authenticated chat endpoints return a friendly setup error instead of crashing the Serverless Function.

## Persistence And Environment

Recommended production variables:

- `DATABASE_URL`: Postgres connection string for durable users and chats.
- `SESSION_SECRET`: long random secret used to sign HttpOnly auth cookies.
- `AUTH_COOKIE_NAME`: optional cookie name override, defaults to `agentic_rag_session`.
- `APP_ENV=production`: enables secure-cookie behavior outside Vercel.

The app keeps optional Azure OpenAI variables unchanged. Wikipedia retrieval needs no API key.

Durable production account/chat storage requires `DATABASE_URL`. `SESSION_SECRET` is still recommended so you control session rotation explicitly; if it is missing while a database is configured, the backend stores a generated signing key in the account database so cookies keep working across serverless restarts. If both `DATABASE_URL` and `SESSION_SECRET` are absent on Vercel, the public shell still renders and signup/login remain usable in temporary mode, with a visible warning in `/api/runtime` and on the auth screen. Set both variables before relying on accounts for real users.

## Auth Troubleshooting

- If the Signup button is disabled, check that the email is valid, the password is at least 8 characters, and the confirmation password matches.
- If signup/login returns a setup error, open `/api/runtime` and check `auth.setup_warning`, `persistence.error`, and `persistence.reachable`.
- If `DATABASE_URL` is missing on Vercel, accounts are temporary and may disappear after a cold start.
- If `SESSION_SECRET` is missing on Vercel but `DATABASE_URL` is configured, the app uses a database-backed generated signing key. Add `SESSION_SECRET` when you want to rotate or pin that key yourself.
- If the database is configured but unreachable, auth endpoints return JSON errors instead of raw stack traces.

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
5. Add `DATABASE_URL` and `SESSION_SECRET` in the Vercel project settings for production account-backed chats.
6. Deploy.

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
- `/api/auth/signup`, `/api/auth/login`, `/api/auth/logout`, and `/api/auth/me` manage accounts and sessions.
- `/api/chat` answers questions with finalized answer, citations, retrieved chunks, guardrails, checkpoints, and pipeline graph data.
- `/api/query` is an alias for `/api/chat`.
- `/api/chat/history` lists the authenticated user's saved chat sessions.
- `/api/chat/history/{session_id}` returns one authenticated user chat transcript.
- `/api/chat/history/{session_id}/clone` clones an authenticated user chat.
- `/api/chats` lists and creates authenticated user chat threads.
- `/api/chats/{session_id}` retrieves, updates, or deletes one authenticated user thread.
- `/api/chats/{session_id}/messages` appends a prompt to an existing user thread and returns the generated answer.
- `/api/chats/{session_id}/messages/{message_id}` edits a user prompt, branches from that turn, and regenerates the answer.
- `/api/chats/{session_id}/regenerate` regenerates the latest assistant response in a user thread.
- `/api/evaluate` runs the bundled golden-set evaluation.

## Local Smoke Run

```cmd
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

Then open `http://127.0.0.1:8000`.

Useful availability checks:

```cmd
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/runtime
```

`/api/runtime` reports whether auth is configured, whether a database URL is present, and whether the database health check is reachable without exposing secret values.
