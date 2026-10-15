import json
from pathlib import Path
from unittest.mock import Mock, patch

from llm_service import ChatMessage, LLMService
from voice_assistant import VoiceAssistant


def test_generate_stream_yields_chunks_and_commits_history_once() -> None:
    response = Mock()
    response.iter_lines.return_value = [
        json.dumps({"response": "Hello"}),
        json.dumps({"response": " from"}),
        json.dumps({"response": " Ollama"}),
        json.dumps({"done": True}),
    ]

    with patch("llm_service.requests.post", return_value=response) as post:
        service = LLMService()
        chunks = list(service.generate_stream("Hi"))

    assert chunks == ["Hello", " from", " Ollama"]
    assert service.history == [
        ChatMessage("user", "Hi"),
        ChatMessage("assistant", "Hello from Ollama"),
    ]
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["stream"] is True
    assert post.call_args.kwargs["json"]["options"] == {"num_predict": 180}
    response.close.assert_called_once()


def test_detailed_requests_use_the_configured_longer_generation_limit() -> None:
    service = LLMService(num_predict=180, detailed_num_predict=360)

    assert service._num_predict_for("Give me a detailed explanation.") == 360
    assert service._num_predict_for("What time is it?") == 180


def test_non_streaming_generation_uses_the_configured_limit() -> None:
    response = Mock()
    response.json.return_value = {"response": "Concise answer."}

    with patch("llm_service.requests.post", return_value=response) as post:
        answer, _ = LLMService(num_predict=120).generate("What is Python?")

    assert answer == "Concise answer."
    assert post.call_args.kwargs["json"]["options"] == {"num_predict": 120}


def test_process_audio_stream_yields_partial_text_then_final_result(tmp_path: Path) -> None:
    stt = Mock()
    stt.transcribe.return_value = ("Recognized input", 0.1)
    llm = Mock()
    llm.generate_stream.return_value = iter(["First", " response"])
    output_path = tmp_path / "response.wav"

    with patch("voice_assistant.synthesize_speech", return_value=output_path) as synthesize:
        updates = list(
            VoiceAssistant(stt_service=stt, llm_service=llm).process_audio_stream(
                tmp_path / "input.wav",
                output_path,
            )
        )

    assert [(transcript, response, result is None) for transcript, response, result in updates] == [
        ("Recognized input", "First", True),
        ("Recognized input", "First response", True),
        ("Recognized input", "First response", False),
    ]
    assert updates[-1][2] is not None
    assert updates[-1][2].response == "First response"
    synthesize.assert_called_once_with("First response", output_path)
