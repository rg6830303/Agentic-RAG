# Troubleshooting

## Vercel Cannot Find A Python Entrypoint

The deployment entrypoint is `app.py`, and it exports a FastAPI instance named `app`. Confirm `pyproject.toml` contains:

```toml
[project.scripts]
app = "app:app"
```

## Unmatched `functions` Pattern

`vercel.json` intentionally does not define a `functions` block. Vercel's FastAPI detector owns the root `app.py` entrypoint.

## `/api/chat` Uses Local Answers

This is expected when Azure OpenAI environment variables are not configured. Add these in Vercel Project Settings to enable model synthesis:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

## Corpus Is Empty

The Vercel app reads from `data/sample_corpus`. Make sure that folder is committed and not ignored by `.vercelignore`.
