# Windows Run

## Prerequisites

- Windows CMD
- Python 3.13 or compatible local Python installation
- Azure OpenAI settings provided through either:
  - a valid `.env` in the repository root, or
  - `.streamlit/secrets.toml` for local Streamlit secret loading

## Commands

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt
python -m streamlit run streamlit_app.py
```

For this repository, `requirements.txt` is the slim Vercel API dependency set. Use `requirements-local.txt` when running Streamlit.

You can also use:

```cmd
run_app.cmd
```

## Notes

- The app writes local state to `artifacts/`
- Uploaded files are stored under `data/uploads/`
- Streamlit pages are available from the sidebar
- For Streamlit Cloud, paste the values from `secrets.toml` into the app's `Secrets` settings instead of uploading `.env`
