from __future__ import annotations

import os
import platform
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


SERVICE_NAME = "Advanced Agentic RAG"
SERVICE_VERSION = "0.1.0"
VERCEL_DEPLOYMENT = bool(os.getenv("VERCEL"))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12_000)
    system_prompt: str = Field(
        default="You are a concise assistant for the Advanced Agentic RAG project.",
        max_length=4_000,
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=700, ge=1, le=4_000)


class ChatResponse(BaseModel):
    answer: str
    provider: str


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description=(
        "Vercel-compatible API entrypoint. The full Streamlit RAG UI is a "
        "local/Streamlit-hosted app because it depends on a writable local "
        "SQLite, FAISS, BM25, and artifact workspace."
    ),
)


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _azure_config() -> dict[str, str]:
    return {
        "endpoint": _env_value("AZURE_OPENAI_ENDPOINT").rstrip("/"),
        "api_version": _env_value("AZURE_OPENAI_API_VERSION"),
        "api_key": _env_value("AZURE_OPENAI_API_KEY"),
        "chat_deployment": _env_value("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "embeddings_deployment": _env_value("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    }


def _azure_status() -> dict[str, Any]:
    config = _azure_config()
    endpoint_host = (
        config["endpoint"]
        .replace("https://", "")
        .replace("http://", "")
        .strip("/")
    )
    return {
        "endpoint_host": endpoint_host,
        "api_version_configured": bool(config["api_version"]),
        "chat_configured": bool(
            config["endpoint"]
            and config["api_version"]
            and config["api_key"]
            and config["chat_deployment"]
        ),
        "embeddings_configured": bool(
            config["endpoint"]
            and config["api_version"]
            and config["api_key"]
            and config["embeddings_deployment"]
        ),
    }


def _azure_chat_url(config: dict[str, str]) -> str:
    return (
        f"{config['endpoint']}/openai/deployments/"
        f"{config['chat_deployment']}/chat/completions"
        f"?api-version={config['api_version']}"
    )


@app.get("/")
def read_root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "deployment": "vercel" if VERCEL_DEPLOYMENT else "local",
        "docs": "/docs",
        "health": "/api/health",
        "runtime": "/api/runtime",
        "chat": "/api/chat",
        "streamlit_ui": (
            "Run locally with `python -m streamlit run streamlit_app.py`; "
            "Streamlit is not a Vercel serverless runtime."
        ),
    }


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/api/runtime")
def runtime_info() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vercel": VERCEL_DEPLOYMENT,
        "azure_openai": _azure_status(),
        "serverless_note": (
            "The disk-backed RAG workspace is intentionally kept out of this "
            "Vercel function. Use the Streamlit app for ingestion, indexing, "
            "FAISS, BM25, checkpoints, and evaluations."
        ),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_completion(payload: ChatRequest) -> ChatResponse:
    config = _azure_config()
    if not _azure_status()["chat_configured"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Azure OpenAI chat is not configured. Set "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_CHAT_DEPLOYMENT in "
                "Vercel Project Settings."
            ),
        )

    try:
        response = requests.post(
            _azure_chat_url(config),
            headers={
                "api-key": config["api_key"],
                "Content-Type": "application/json",
            },
            json={
                "messages": [
                    {"role": "system", "content": payload.system_prompt},
                    {"role": "user", "content": payload.message},
                ],
                "temperature": payload.temperature,
                "max_tokens": payload.max_tokens,
            },
            timeout=25,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Azure OpenAI request failed: {exc}",
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Azure OpenAI returned an error: {response.text[:500]}",
        )

    body = response.json()
    choices = body.get("choices", [])
    if not choices:
        raise HTTPException(
            status_code=502,
            detail="Azure OpenAI returned no choices.",
        )
    answer = str(choices[0].get("message", {}).get("content", "")).strip()
    return ChatResponse(answer=answer, provider="azure_openai")
