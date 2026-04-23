from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from src.config.settings import AppSettings


class ProviderError(RuntimeError):
    """Raised when the Azure OpenAI provider cannot fulfill a request."""


@dataclass(slots=True)
class AzureOpenAIProvider:
    settings: AppSettings

    def _headers(self) -> dict[str, str]:
        return {
            "api-key": self.settings.azure_api_key,
            "Content-Type": "application/json",
        }

    def _url(self, deployment: str, path: str) -> str:
        endpoint = self.settings.azure_endpoint.rstrip("/")
        return (
            f"{endpoint}/openai/deployments/{deployment}/{path}"
            f"?api-version={self.settings.azure_api_version}"
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.embeddings_available:
            raise ProviderError("Embeddings deployment is not configured.")
        cleaned = [text.strip() or " " for text in texts]
        vectors: list[list[float]] = []
        batch_size = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            payload = {"input": batch}
            try:
                response = requests.post(
                    self._url(self.settings.azure_embeddings_deployment, "embeddings"),
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.embedding_timeout_seconds,
                )
            except requests.RequestException as exc:
                raise ProviderError(f"Azure embeddings request failed: {exc}") from exc
            if response.status_code >= 400:
                raise ProviderError(
                    f"Azure embeddings request failed with status {response.status_code}: {response.text[:300]}"
                )
            body = response.json()
            ordered = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
            vectors.extend(item.get("embedding", []) for item in ordered)
        return vectors

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 700,
    ) -> str:
        if not self.settings.chat_available:
            raise ProviderError("Chat deployment is not configured.")
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(
                self._url(self.settings.azure_chat_deployment, "chat/completions"),
                headers=self._headers(),
                json=payload,
                timeout=self.settings.chat_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Azure chat request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"Azure chat request failed with status {response.status_code}: {response.text[:300]}"
            )
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise ProviderError("Azure chat response did not include any choices.")
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()
