# Local Run

This repository is configured for Vercel deployment. A local smoke run is still available for checking the FastAPI UI before pushing.

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

Azure OpenAI settings can be provided through environment variables or a local `.env` file when using the reusable Python services under `src/`.
