"""Independent command-line test for the local Piper text-to-speech service."""

from __future__ import annotations

from pathlib import Path

from tts_service import synthesize_speech


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "test_tts.wav"


def main() -> None:
    output_path = synthesize_speech(
        "Hello, this is a local voice assistant.",
        OUTPUT_PATH,
    )
    print(f"Generated WAV: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
