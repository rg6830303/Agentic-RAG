# Windows Run

## Prerequisites

- Windows CMD
- Python 3.13 or compatible local Python installation
- A valid `.env` in the repository root with Azure OpenAI settings

## Commands

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

You can also use:

```cmd
run_app.cmd
```

## Notes

- The app writes local state to `artifacts/`
- Uploaded files are stored under `data/uploads/`
- Streamlit pages are available from the sidebar
