"""End-to-end local voice assistant: STT -> Ollama -> Piper."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from llm_service import LLMService, LLMServiceError
from stt_service import SpeechToTextService
from tts_service import TTSServiceError, synthesize_speech


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "assistant_response.wav"


class VoiceAssistantError(RuntimeError):
    """Raised for an expected end-to-end pipeline failure."""


@dataclass(frozen=True)
class PipelineResult:
    transcript: str
    response: str
    audio_path: Path
    stt_latency: float
    llm_latency: float
    tts_latency: float
    total_latency: float


class VoiceAssistant:
    """Coordinate local STT, LLM, and TTS services across multiple turns."""

    def __init__(
        self,
        stt_service: SpeechToTextService | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.stt_service = stt_service or SpeechToTextService()
        self.llm_service = llm_service or LLMService()

    def process_audio(self, audio_path: str | Path, output_path: str | Path = DEFAULT_OUTPUT) -> PipelineResult:
        total_start = time.perf_counter()
        try:
            transcript, stt_latency = self.stt_service.transcribe(audio_path)
        except FileNotFoundError as error:
            raise VoiceAssistantError(str(error)) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise VoiceAssistantError(f"Speech-to-text failed: {error}") from error

        if not transcript:
            raise VoiceAssistantError("Speech-to-text produced an empty transcript")

        try:
            response, llm_latency = self.llm_service.generate(transcript)
        except (LLMServiceError, ValueError) as error:
            raise VoiceAssistantError(str(error)) from error

        tts_start = time.perf_counter()
        try:
            audio_output = synthesize_speech(response, output_path)
        except (FileNotFoundError, TTSServiceError, ValueError) as error:
            raise VoiceAssistantError(f"Text-to-speech failed: {error}") from error

        return PipelineResult(
            transcript=transcript,
            response=response,
            audio_path=audio_output,
            stt_latency=stt_latency,
            llm_latency=llm_latency,
            tts_latency=time.perf_counter() - tts_start,
            total_latency=time.perf_counter() - total_start,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local STT -> LLM -> TTS pipeline.")
    parser.add_argument("audio_file", type=Path, help="Input audio file to transcribe")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output WAV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        result = VoiceAssistant().process_audio(args.audio_file, args.output)
    except VoiceAssistantError as error:
        parser.error(str(error))

    print("\nTranscript:")
    print(result.transcript)
    print("\nAssistant response:")
    print(result.response)
    print(f"\nResponse audio: {result.audio_path}")
    print(f"STT latency: {result.stt_latency:.2f} seconds")
    print(f"LLM latency: {result.llm_latency:.2f} seconds")
    print(f"TTS latency: {result.tts_latency:.2f} seconds")
    print(f"Total latency: {result.total_latency:.2f} seconds")


if __name__ == "__main__":
    main()
