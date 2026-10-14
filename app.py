"""Gradio interface for the local STT -> LLM -> TTS voice assistant."""

from __future__ import annotations

import threading
import tempfile
import traceback
import wave
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np

from llm_service import LLMService
from stt_service import SpeechToTextService
from voice_assistant import PROJECT_ROOT, VoiceAssistant, VoiceAssistantError


OUTPUT_DIR = PROJECT_ROOT / "outputs"
_stt_service: SpeechToTextService | None = None
_stt_lock = threading.Lock()


def _get_shared_stt_service() -> SpeechToTextService:
    """Load the memory-heavy Whisper model once for the application process."""
    global _stt_service
    if _stt_service is None:
        with _stt_lock:
            if _stt_service is None:
                _stt_service = SpeechToTextService()
    return _stt_service


def _history_messages(assistant: VoiceAssistant | None) -> list[dict[str, str]]:
    if assistant is None:
        return []
    return [
        {"role": message.role, "content": message.content}
        for message in assistant.llm_service.history
    ]


def _audio_duration(audio_file: Path) -> float | None:
    try:
        with wave.open(str(audio_file), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            return wav_file.getnframes() / frame_rate if frame_rate else None
    except (OSError, wave.Error):
        return None


def _audio_input_to_file(
    audio_input: str | Path | tuple[int, np.ndarray],
) -> tuple[Path, bool, float | None]:
    """Normalize Gradio filepath or NumPy audio input to a temporary WAV path."""
    if isinstance(audio_input, (str, Path)):
        audio_file = Path(audio_input)
        if not audio_file.is_file() or audio_file.stat().st_size == 0:
            raise ValueError(f"the recorded audio file is missing or empty: {audio_file}")
        return audio_file, False, _audio_duration(audio_file)

    if not isinstance(audio_input, tuple) or len(audio_input) != 2:
        raise ValueError(
            "Gradio returned an unsupported audio value; expected a file path "
            "or (sample_rate, audio_array)"
        )

    sample_rate, audio_array = audio_input
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise ValueError("audio sample rate is missing or invalid")
    if not isinstance(audio_array, np.ndarray) or audio_array.size == 0:
        raise ValueError("audio array is missing or empty")
    if audio_array.ndim not in (1, 2):
        raise ValueError("audio array must contain one or two dimensions")

    samples = np.asarray(audio_array)
    if samples.ndim == 2 and samples.shape[1] > 2:
        raise ValueError("audio array has more than two channels")
    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0) * 32767.0
    samples = np.clip(samples, -32768, 32767).astype(np.int16)

    temporary_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    channels = samples.shape[1] if samples.ndim == 2 else 1
    with wave.open(str(temporary_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(samples.tobytes())
    return temporary_path, True, samples.shape[0] / float(sample_rate)


def process_turn(
    audio_path: str | Path | tuple[int, np.ndarray] | None,
    assistant: VoiceAssistant | None,
) -> tuple[str, str, str | None, str, list[dict[str, str]], VoiceAssistant | None]:
    """Process one audio turn and return all UI values."""
    if audio_path is None:
        message = "Error: no audio was received. Record speech or upload a WAV file first."
        return message, "", None, message, _history_messages(assistant), assistant

    temporary_audio = False
    try:
        audio_file, temporary_audio, duration = _audio_input_to_file(audio_path)
        sample_rate = None
        if isinstance(audio_path, tuple):
            sample_rate = audio_path[0]
        print(
            f"Received audio type={type(audio_path).__name__}, "
            f"sample_rate={sample_rate or 'from file'}, "
            f"duration={duration:.2f}s" if duration is not None
            else f"Received audio type={type(audio_path).__name__}, "
            f"sample_rate={sample_rate or 'from file'}, duration=unknown",
            flush=True,
        )
    except (OSError, ValueError) as error:
        message = f"Error: {error}"
        print(message, flush=True)
        return message, "", None, message, _history_messages(assistant), assistant

    try:
        if assistant is None:
            assistant = VoiceAssistant(
                stt_service=_get_shared_stt_service(),
                llm_service=LLMService(),
            )

        output_path = OUTPUT_DIR / f"response_{uuid4().hex}.wav"
        try:
            result = assistant.process_audio(audio_file, output_path)
        except VoiceAssistantError as error:
            traceback.print_exc()
            message = f"Error: {error}"
            return message, "", None, message, _history_messages(assistant), assistant
        except Exception as error:
            traceback.print_exc()
            message = f"Unexpected processing error: {error}"
            return message, "", None, message, _history_messages(assistant), assistant

        metrics = (
            f"STT: {result.stt_latency:.2f}s | "
            f"LLM: {result.llm_latency:.2f}s | "
            f"TTS: {result.tts_latency:.2f}s | "
            f"Total: {result.total_latency:.2f}s"
        )
        return (
            result.transcript,
            result.response,
            str(result.audio_path),
            metrics,
            _history_messages(assistant),
            assistant,
        )
    finally:
        if temporary_audio:
            audio_file.unlink(missing_ok=True)


def clear_conversation() -> tuple[
    None,
    str,
    str,
    None,
    str,
    list[dict[str, str]],
    None,
]:
    """Reset the current browser session's conversation and displayed results."""
    return None, "", "", None, "Conversation cleared.", [], None


def clear_audio() -> None:
    """Reset only the current audio input."""
    return None


with gr.Blocks(title="Local Voice AI Assistant") as demo:
    gr.Markdown(
        "# Local Voice AI Assistant\n"
        "Private local inference with Faster-Whisper, Ollama Qwen, and Piper."
    )

    assistant_state = gr.State(value=None)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                value=None,
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                streaming=False,
                recording=False,
                label="Speak or upload audio",
            )
            with gr.Row():
                process_button = gr.Button("Process", variant="primary")
                clear_audio_button = gr.Button("Clear audio")
                clear_button = gr.Button("Clear conversation")
        with gr.Column():
            transcript_output = gr.Textbox(label="Recognized transcript", lines=4)
            response_output = gr.Textbox(label="Assistant response", lines=5)
            audio_output = gr.Audio(label="Response audio", type="filepath")
            latency_output = gr.Markdown("Record or upload audio to begin.")

    history_output = gr.Chatbot(
        label="Conversation history",
        value=[],
    )

    process_button.click(
        process_turn,
        inputs=[audio_input, assistant_state],
        outputs=[
            transcript_output,
            response_output,
            audio_output,
            latency_output,
            history_output,
            assistant_state,
        ],
    )
    clear_audio_button.click(
        clear_audio,
        outputs=audio_input,
    )
    clear_button.click(
        clear_conversation,
        outputs=[
            audio_input,
            transcript_output,
            response_output,
            audio_output,
            latency_output,
            history_output,
            assistant_state,
        ],
    )


if __name__ == "__main__":
    demo.launch()
