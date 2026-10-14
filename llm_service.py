"""Local language-model service backed by Ollama."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:1.5b"


class LLMServiceError(RuntimeError):
    """Raised when Ollama cannot generate a response."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMService:
    """Generate responses locally while retaining conversation history."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 180,
    ) -> None:
        self.model = model
        self.url = url
        self.timeout = timeout
        self.history: list[ChatMessage] = []

    def generate(self, user_text: str) -> tuple[str, float]:
        """Generate an assistant response and return it with its latency."""
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must contain at least one non-whitespace character")

        user_message = user_text.strip()
        prompt = self._build_prompt(user_message)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        start = time.perf_counter()
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError as error:
            raise LLMServiceError(
                "Ollama is unavailable. Start Ollama and verify its local service is running."
            ) from error
        except requests.exceptions.Timeout as error:
            raise LLMServiceError(f"Ollama timed out after {self.timeout:.0f} seconds") from error
        except requests.exceptions.RequestException as error:
            raise LLMServiceError(f"Ollama request failed: {error}") from error
        except ValueError as error:
            raise LLMServiceError("Ollama returned invalid JSON") from error

        answer = result.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise LLMServiceError("Ollama returned an empty response")

        answer = answer.strip()
        self.history.extend(
            [
                ChatMessage("user", user_message),
                ChatMessage("assistant", answer),
            ]
        )
        return answer, time.perf_counter() - start

    def _build_prompt(self, user_text: str) -> str:
        lines = [
            "You are a helpful local voice assistant. Respond clearly and concisely.",
            "",
        ]
        for message in self.history:
            lines.append(f"{message.role.title()}: {message.content}")
        lines.append(f"User: {user_text}")
        lines.append("Assistant:")
        return "\n".join(lines)
