"""Local speech-to-text service backed by Faster-Whisper."""

from __future__ import annotations

import time
from pathlib import Path

from faster_whisper import WhisperModel


DEFAULT_MODEL_SIZE = "base"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


class SpeechToTextService:
    """Transcribe audio locally with a CPU-friendly Faster-Whisper model."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str | Path) -> tuple[str, float]:
        """Return the transcription and transcription latency in seconds."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        start = time.perf_counter()
        segments, info = self.model.transcribe(
            str(path),
            language=None,
            beam_size=5,
            vad_filter=True,
        )
        print(
            f"Detected language: {info.language} "
            f"(probability: {info.language_probability:.2f})",
            flush=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        latency = time.perf_counter() - start
        return text, latency
