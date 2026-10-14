"""Independent command-line test for the local speech-to-text service."""

from __future__ import annotations

import argparse
from pathlib import Path

from stt_service import SpeechToTextService


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Faster-Whisper.")
    parser.add_argument("audio_file", type=Path, help="Path to a WAV, MP3, M4A, or FLAC file")
    args = parser.parse_args()

    print("Loading Faster-Whisper tiny model (CPU INT8)...")
    service = SpeechToTextService()
    text, latency = service.transcribe(args.audio_file)

    print("\nTranscription:")
    print(text or "[No speech detected]")
    print(f"\nTranscription time: {latency:.2f} seconds")


if __name__ == "__main__":
    main()
