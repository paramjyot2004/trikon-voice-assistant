"""Local text-to-speech service backed by the Piper executable."""

from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PIPER_EXECUTABLE = PROJECT_ROOT.parent / "tools" / "piper" / "piper" / "piper.exe"
DEFAULT_MODEL = PROJECT_ROOT.parent / "models" / "piper" / "en_US-lessac-medium.onnx"


class TTSServiceError(RuntimeError):
    """Raised when Piper cannot synthesize valid speech output."""


def _configured_path(environment_name: str, default: Path) -> Path:
    configured = os.environ.get(environment_name)
    return Path(configured).expanduser() if configured else default


def _validate_assets(executable: Path, model: Path, config: Path) -> None:
    missing = [
        f"{label}: {path}"
        for label, path in (
            ("Piper executable", executable),
            ("Piper model", model),
            ("Piper model configuration", config),
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing Piper asset(s): " + "; ".join(missing))


def synthesize_speech(text: str, output_path: str | Path) -> Path:
    """Generate a WAV file from text using the local Piper voice model."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must contain at least one non-whitespace character")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    executable = _configured_path("PIPER_EXECUTABLE", DEFAULT_PIPER_EXECUTABLE)
    model = _configured_path("PIPER_MODEL", DEFAULT_MODEL)
    config = Path(f"{model}.json")
    _validate_assets(executable, model, config)

    command = [
        str(executable),
        "--model",
        str(model),
        "--config",
        str(config),
        "--output_file",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            input=text.strip() + "\n",
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            cwd=executable.parent,
        )
    except OSError as error:
        raise TTSServiceError(f"Could not start Piper: {error}") from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        message = f"Piper failed with exit code {error.returncode}"
        raise TTSServiceError(f"{message}: {details}" if details else message) from error

    if not output.is_file() or output.stat().st_size == 0:
        details = completed.stderr.strip()
        raise TTSServiceError(
            f"Piper did not produce a valid WAV file at {output}"
            + (f": {details}" if details else "")
        )

    try:
        with wave.open(str(output), "rb") as wav_file:
            if wav_file.getnframes() == 0:
                raise TTSServiceError(f"Piper produced an empty WAV file at {output}")
    except (wave.Error, EOFError) as error:
        raise TTSServiceError(f"Piper produced an invalid WAV file at {output}") from error

    return output
