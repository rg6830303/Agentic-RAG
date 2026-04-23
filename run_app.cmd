@echo off
setlocal
if exist .venv\Scripts\python.exe (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)
%PYTHON_EXE% -m streamlit run app.py
