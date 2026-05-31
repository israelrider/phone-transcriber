# phone-transcriber

Convert a Hebrew phone-call MP3 (or WAV / M4A / OGG) to a clean text file.

**Two-stage pipeline:**

| Stage | What happens | Where it runs |
|-------|-------------|---------------|
| 1 — Transcription | [OpenAI Whisper](https://github.com/openai/whisper) converts audio → raw Hebrew text | **Local** (GPU or CPU) |
| 2 — Cleanup | An Ollama Cloud language model fixes punctuation, adds speaker labels, corrects errors | **Ollama Cloud** (remote) |

---

## Prerequisites

### 1. Python ≥ 3.11

```
python --version
```

### 2. FFmpeg

Whisper requires FFmpeg on your `PATH`.

**Windows (recommended — winget):**
```
winget install Gyan.FFmpeg
```
Then open a new terminal and verify:
```
ffmpeg -version
```

**Alternative — Chocolatey:**
```
choco install ffmpeg
```

### 3. Ollama Cloud account & API key

1. Go to <https://cloud.ollama.com> and sign in (or create an account).
2. Navigate to **Settings → API Keys** and create a new key.
3. Pull a model that supports Hebrew — recommended choices:

   | Model | Notes |
   |-------|-------|
   | `llama3.1:8b` | Best Hebrew quality in this size range |
   | `mistral:7b` | Solid multilingual alternative |
   | `qwen2.5:7b` | Strong multilingual, very token-efficient |

   In the Ollama Cloud web UI, click **Pull** next to your chosen model.

---

## Installation

```
cd C:\Users\user\Documents\localhost\python\phone-transcriber
pip install -e .
```

This installs the `phone-transcriber` CLI and all Python dependencies
(`openai-whisper`, `httpx`, `python-dotenv`).

> **First run note:** Whisper downloads its model weights automatically on
> first use (`~medium` ≈ 1.5 GB). Subsequent runs use the cached copy.

---

## Configuration

Copy the example env file and edit it:

```
copy .env.example .env
```

Open `.env` in any editor and set:

```dotenv
# Your Ollama Cloud API key
OPENAI_API_KEY=ollama_xxxxxxxxxxxxxxxxxxxxxxxx

# Ollama Cloud endpoint (leave as-is unless self-hosting)
OPENAI_BASE_URL=https://cloud.ollama.com

# The model you pulled in the Ollama Cloud UI
OPENAI_MODEL=llama3.1:8b
```

All three variables are **required** unless you use `--skip-cleanup`.

---

## Usage

### Basic — transcribe a file

```
phone-transcriber transcribe --input "C:\path\to\call.mp3"
```

Output is saved to `artifacts\<filename>_<timestamp>.txt` automatically.

### Specify output path

```
phone-transcriber transcribe --input call.mp3 --output transcript.txt
```

### Use a larger Whisper model (better accuracy, slower)

```
phone-transcriber transcribe --input call.mp3 --whisper-model large-v3
```

Available Whisper sizes (smallest → largest / fastest → slowest):

| Model | VRAM | Speed | Hebrew accuracy |
|-------|------|-------|-----------------|
| `tiny` | ~1 GB | fastest | fair |
| `base` | ~1 GB | fast | fair |
| `small` | ~2 GB | fast | good |
| `medium` | ~5 GB | moderate | **very good** ← default |
| `large-v3` | ~10 GB | slow | excellent |

### Skip the Ollama cleanup step (raw Whisper output only)

Useful for testing or when Ollama Cloud is unavailable.

```
phone-transcriber transcribe --input call.mp3 --skip-cleanup
```

### Dry run — see what would happen without calling anything

```
phone-transcriber transcribe --input call.mp3 --dry-run
```

### Increase timeout for long calls

```
phone-transcriber transcribe --input long_call.mp3 --timeout-s 600
```

### Set Whisper model via environment variable

Add to `.env`:
```dotenv
WHISPER_MODEL=large-v3
```
Then just run:
```
phone-transcriber transcribe --input call.mp3
```

---

## Output format

The cleaned transcript is plain UTF-8 text. When speakers are
distinguishable, the Ollama model formats the dialogue like:

```
דובר 1: שלום, מה שלומך?
דובר 2: בסדר גמור, תודה. מה נשמע?
דובר 1: רציתי לשאול אותך לגבי הפגישה של מחר...
```

If speakers cannot be reliably separated, the text appears as continuous
paragraphs.

---

## Environment variables reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Ollama Cloud API key |
| `OPENAI_BASE_URL` | Yes | API base URL (`https://cloud.ollama.com`) |
| `OPENAI_MODEL` | Yes | Model name as shown in Ollama Cloud |
| `WHISPER_MODEL` | No | Whisper model size (default: `medium`) |

Variables are loaded from `.env` in the working directory, or can be set
in the shell environment directly.

---

## Troubleshooting

**`ffmpeg not found` / `FileNotFoundError`**
Install FFmpeg and open a new terminal so the updated `PATH` takes effect.

**`OPENAI_API_KEY is not set`**
Make sure `.env` exists in the directory where you run the command,
or export the variable in your shell.

**Model not found on Ollama Cloud**
Open <https://cloud.ollama.com>, go to **Models**, and pull the model
specified in `OPENAI_MODEL`.

**Out of memory during Whisper**
Use a smaller Whisper model: `--whisper-model small` or `--whisper-model base`.

**Transcription is inaccurate**
Switch to a larger Whisper model: `--whisper-model large-v3`.
For very noisy phone audio, `large-v3` makes a significant difference.

**Hebrew text appears garbled / wrong direction in terminal**
This is a terminal rendering issue, not a data issue. Open the `.txt` file
in Notepad, VS Code, or any editor with RTL support — the content is correct.

---

## Project structure

```
phone-transcriber/
├── .env.example              ← copy to .env and fill in credentials
├── pyproject.toml
├── README.md
├── artifacts/                ← output transcripts land here
└── phone_transcriber/
    ├── cli.py                ← argument parsing and orchestration
    ├── transcriber.py        ← Whisper audio-to-text (Stage 1)
    ├── ollama_client.py      ← Ollama Cloud HTTP client (Stage 2)
    └── prompts.py            ← system & user prompts for the LLM
```
