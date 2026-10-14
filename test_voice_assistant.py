"""Run the integrated local STT -> LLM -> TTS pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voice_assistant import DEFAULT_OUTPUT, VoiceAssistant, VoiceAssistantError


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "test.wav",
    PROJECT_ROOT.parent / "outputs" / "test.wav",
)


def default_audio_path() -> Path:
    for candidate in DEFAULT_AUDIO_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_AUDIO_CANDIDATES[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local STT -> Ollama Qwen -> Piper pipeline."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        default=default_audio_path(),
        help="Input audio path (defaults to an existing test audio file when available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Generated WAV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.audio_file.is_file():
        print(f"Error: audio file not found: {args.audio_file}", file=sys.stderr)
        return 2

    try:
        result = VoiceAssistant().process_audio(args.audio_file, args.output)
    except VoiceAssistantError as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return 1

    print("\nRecognized transcript:")
    print(result.transcript)
    print("\nAssistant response:")
    print(result.response)
    print(f"\nGenerated WAV: {result.audio_path}")
    print(f"STT latency: {result.stt_latency:.2f} seconds")
    print(f"LLM latency: {result.llm_latency:.2f} seconds")
    print(f"TTS latency: {result.tts_latency:.2f} seconds")
    print(f"Total latency: {result.total_latency:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
