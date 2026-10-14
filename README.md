# Trikon Software Labs Voice AI Assistant

An end-to-end local voice assistant using open-weight/local inference components:

```text
Microphone or WAV
    -> Gradio
    -> Faster-Whisper STT
    -> Ollama + Qwen
    -> Piper TTS
    -> Response WAV playback
```

The project is designed for Windows 11 and a laptop with 8 GB RAM. The initial speech-to-text configuration uses the Faster-Whisper `tiny` model with CPU INT8 to keep memory use practical.

## Features

- Microphone recording and WAV upload through Gradio
- Local speech-to-text with Faster-Whisper
- Local response generation with Ollama and Qwen
- Local text-to-speech with Piper
- Per-turn STT, LLM, TTS, and total latency measurements
- Session conversation history across multiple turns
- Clear-audio and clear-conversation controls
- Input validation and visible error messages
- No paid or hosted AI inference APIs

## Prerequisites

- Windows 11
- Python installed and available to create a virtual environment
- Ollama installed and running locally
- The Ollama model `qwen2.5:1.5b`
- The existing Piper executable and voice assets:
  - `..\tools\piper\piper\piper.exe`
  - `..\models\piper\en_US-lessac-medium.onnx`
  - `..\models\piper\en_US-lessac-medium.onnx.json`
- A working microphone is optional if WAV upload is used

The Piper assets in the current setup are stored one directory above this project, under `D:\voice-assistant\models\piper` and `D:\voice-assistant\tools\piper\piper`. `tts_service.py` resolves those paths relative to the project directory. Override them with `PIPER_EXECUTABLE` and `PIPER_MODEL` if your layout differs.

## Models and configuration

### Speech-to-text

`stt_service.py` uses:

```text
Model: tiny
Device: CPU
Compute type: int8
```

Faster-Whisper downloads its model files to the local Hugging Face cache on first use if they are not already cached.

### Language model

`llm_service.py` calls the local Ollama endpoint:

```text
http://localhost:11434/api/generate
```

Configured model:

```text
qwen2.5:1.5b
```

### Text-to-speech

The current Piper voice metadata identifies:

```text
Voice: en_US-lessac-medium
Language: English (United States)
Dataset/voice: Lessac
Quality: medium
Sample rate: 22050 Hz
Speakers: 1
```

## Windows setup

From PowerShell:

```powershell
cd D:\voice-assistant\trikon-voice-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the project interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not commit `.venv` or downloaded model files.

## Ollama setup

Install Ollama from its official Windows installer, start the Ollama service, and make sure the model is available:

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

The application uses Ollama through its local HTTP API. Confirm Ollama is running before starting the application.

## Piper setup

Place or preserve the Piper assets in the layout expected by `tts_service.py`:

```text
D:\voice-assistant\
├── models\piper\en_US-lessac-medium.onnx
├── models\piper\en_US-lessac-medium.onnx.json
└── tools\piper\piper\piper.exe
```

The project does not need a hosted TTS service. Piper runs locally through the existing Windows executable.

## Running the application

Start the Gradio interface:

```powershell
cd D:\voice-assistant\trikon-voice-assistant
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed by Gradio, normally:

```text
http://127.0.0.1:7860
```

Allow microphone permission when recording. Alternatively, upload a WAV file and click **Process**. The response text, generated WAV, conversation history, and latency values are shown in the interface.

## Command-line checks

Run the existing independent component checks:

```powershell
.\.venv\Scripts\python.exe test_llm.py
.\.venv\Scripts\python.exe test_stt.py ..\outputs\test.wav
.\.venv\Scripts\python.exe test_tts.py
```

Run the integrated pipeline:

```powershell
.\.venv\Scripts\python.exe test_voice_assistant.py ..\outputs\test.wav
```

## Testing the application

### Microphone recording

1. Start `app.py` and open its local Gradio URL.
2. Allow microphone access.
3. Click the built-in Record control.
4. Speak for two or three seconds and click Stop.
5. Confirm the recorded audio remains visible.
6. Click **Process**.
7. Verify the transcript, assistant response, response WAV, and four latency values.

### WAV upload

1. Select the upload option in the audio component.
2. Choose a known-good WAV file, such as `..\outputs\test.wav` when it exists.
3. Wait for the audio preview to appear.
4. Click **Process** and verify the same outputs.

### Generated speech

Play the response audio shown by Gradio. The command-line TTS test also writes `outputs\test_tts.wav`; generated audio files are ignored by Git.

### Two-turn conversation memory

1. Process a first turn, for example: “What is artificial intelligence?”
2. Process a related second turn, for example: “Explain it in simpler words.”
3. Confirm both turns appear in the conversation history.
4. Confirm the second response can use the first turn's context.
5. Use **Clear conversation** to reset history. Use **Clear audio** when only the current recording should be removed.

## Troubleshooting

### Ollama unavailable

Start Ollama and verify the model:

```powershell
ollama list
```

The application reports an Ollama connection or timeout error in the UI and prints the traceback in the terminal for unexpected failures.

### No audio received

Make sure recording has stopped and that the waveform remains visible before clicking **Process**. For a controlled check, upload a WAV file instead. The app logs the received input type, sample rate when available, and duration without printing audio contents.

### Piper files not found

Verify the executable, `.onnx`, and matching `.onnx.json` paths. For a different layout, set:

```powershell
$env:PIPER_EXECUTABLE = "C:\path\to\piper.exe"
$env:PIPER_MODEL = "C:\path\to\voice.onnx"
```

### First-run model loading

The first Faster-Whisper run may download and load model files. Subsequent runs use the local cache. The application keeps the Whisper model loaded for the Gradio process rather than loading it for every turn.

### Browser microphone permissions

Check the browser's microphone permission for the local Gradio page. If the component remains in an unexpected state after code changes, stop old Gradio processes and open the newly printed URL in a fresh tab.

## Hardware and latency notes

The tested target is a Windows 11 laptop with 8 GB RAM using CPU INT8 STT. Actual latency depends on CPU speed, model warm-up, prompt length, audio duration, and system load.

Earlier observed end-to-end run (example only, not a guaranteed benchmark):

```text
STT: 1.28 s
LLM: 10.04 s
TTS: 1.22 s
Total: 12.54 s
```

These values are retained as a prior observed run and should not be treated as a fixed performance guarantee.

## Project layout

```text
trikon-voice-assistant/
├── app.py
├── llm_service.py
├── stt_service.py
├── tts_service.py
├── voice_assistant.py
├── test_llm.py
├── test_stt.py
├── test_tts.py
├── test_voice_assistant.py
├── requirements.txt
└── docs/
    └── architecture.md
```
