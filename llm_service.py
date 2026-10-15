"""Local language-model service backed by Ollama."""

from __future__ import annotations

import time
import json
from collections.abc import Iterator
from dataclasses import dataclass

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_NUM_PREDICT = 180
DETAILED_NUM_PREDICT = 360
_DETAIL_REQUEST_MARKERS = (
    "detailed",
    "in detail",
    "step by step",
    "thorough",
    "comprehensive",
    "in-depth",
    "deep dive",
    "elaborate",
)


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
        num_predict: int = DEFAULT_NUM_PREDICT,
        detailed_num_predict: int = DETAILED_NUM_PREDICT,
    ) -> None:
        if num_predict <= 0 or detailed_num_predict <= 0:
            raise ValueError("num_predict limits must be positive")
        if detailed_num_predict < num_predict:
            raise ValueError("detailed_num_predict must be at least num_predict")
        self.model = model
        self.url = url
        self.timeout = timeout
        self.num_predict = num_predict
        self.detailed_num_predict = detailed_num_predict
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
            "options": {"num_predict": self._num_predict_for(user_message)},
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

    def generate_stream(self, user_text: str) -> Iterator[str]:
        """Yield response chunks from Ollama and store history after completion."""
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must contain at least one non-whitespace character")

        user_message = user_text.strip()
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(user_message),
            "stream": True,
            "options": {"num_predict": self._num_predict_for(user_message)},
        }
        chunks: list[str] = []
        response = None
        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except (TypeError, json.JSONDecodeError) as error:
                    raise LLMServiceError("Ollama returned invalid streaming JSON") from error
                if not isinstance(result, dict):
                    raise LLMServiceError("Ollama returned invalid streaming JSON")

                if result.get("error"):
                    raise LLMServiceError(f"Ollama returned an error: {result['error']}")
                chunk = result.get("response")
                if not isinstance(chunk, str):
                    continue
                chunks.append(chunk)
                yield chunk
        except requests.exceptions.ConnectionError as error:
            raise LLMServiceError(
                "Ollama is unavailable. Start Ollama and verify its local service is running."
            ) from error
        except requests.exceptions.Timeout as error:
            raise LLMServiceError(f"Ollama timed out after {self.timeout:.0f} seconds") from error
        except requests.exceptions.RequestException as error:
            raise LLMServiceError(f"Ollama request failed: {error}") from error
        finally:
            if response is not None:
                response.close()

        answer = "".join(chunks).strip()
        if not answer:
            raise LLMServiceError("Ollama returned an empty response")

        self.history.extend(
            [
                ChatMessage("user", user_message),
                ChatMessage("assistant", answer),
            ]
        )

    def _build_prompt(self, user_text: str) -> str:
        lines = [
            "You are a helpful local voice assistant.",
            "For ordinary questions, answer in 3-5 sentences using simple, natural language for speech.",
            "Avoid long introductions, repeated explanations, unnecessary headings, and excessive examples.",
            "If the user explicitly asks for a detailed, thorough, step-by-step, or comprehensive explanation, provide the requested additional detail.",
            "",
        ]
        for message in self.history:
            lines.append(f"{message.role.title()}: {message.content}")
        lines.append(f"User: {user_text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _num_predict_for(self, user_text: str) -> int:
        """Allow extra generation room only when the user explicitly requests detail."""
        normalized_text = user_text.casefold()
        if any(marker in normalized_text for marker in _DETAIL_REQUEST_MARKERS):
            return self.detailed_num_predict
        return self.num_predict
