# phone-transcriber

Transcribe phone-call recordings (Hebrew, Russian, English) to text in the
same language, **entirely on this machine** — no audio leaves the computer.

Engine: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).

| Language | Model | Why |
|----------|-------|-----|
| Hebrew | `ivrit-ai/whisper-large-v3-turbo-ct2` | ivrit.ai's Hebrew fine-tune; much better than stock Whisper on Hebrew |
| Russian, English | `large-v3-turbo` | Stock multilingual Whisper, already strong on both |

Models download automatically on first use (~1.6 GB each) into the Hugging
Face cache (`%USERPROFILE%\.cache\huggingface`).

Output for each recording, in `transcripts\` by default:

- `<name>.txt` — the transcript, one phrase per line
- `<name>.srt` — the same text with timestamps, so you can jump to any line
  in the audio to check it (VLC opens `.srt` next to the audio file)

---

## Setup (once)

Requires Python 3.13. FFmpeg is **not** needed (audio decoding is built in),
though it's handy for converting files.

```powershell
cd C:\Users\user\Documents\localhost\python\phone-transcriber
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[gpu]"
```

The `[gpu]` extra installs the CUDA 12 cuBLAS / cuDNN 9 DLLs needed to run on
the NVIDIA GPU. Without it everything still works, on the CPU.

## Usage

Activate the venv once per terminal:

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
# A file, several files, or a whole folder — language is detected per file
phone-transcriber transcribe "C:\calls\call.amr"
phone-transcriber transcribe "C:\calls"

# Force a language (skips detection)
phone-transcriber transcribe "C:\calls\call.amr" --lang ru
```

Language detection uses the general model, because the Hebrew model can't do
it: it labels every recording as Hebrew, and forcing Russian audio through it
produces Russian written phonetically in Hebrew letters. Only force `--lang`
when you're sure.

Supported input: `.amr .mp3 .m4a .wav .ogg .opus .3gp .aac .flac .wma`.

Files that already have a transcript in the output folder are skipped, so an
interrupted folder run can simply be restarted. Use `--force` to redo them.

| Option | Default | Meaning |
|--------|---------|---------|
| `--lang` | `auto` | `auto` (detect per file), or force `he`, `ru`, `en` |
| `--out-dir` | `transcripts` | Where `.txt` / `.srt` files go |
| `--model` | per language | Any faster-whisper model, e.g. `ivrit-ai/whisper-large-v3-ct2` (slower, possibly more accurate) |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` (GPU if available, falls back to CPU if it fails) |
| `--compute-type` | `int8` | CTranslate2 precision. `int8` fits a 4 GB GPU; Pascal GPUs have no `float16` |
| `--force` | off | Re-transcribe files that already have output |

### Speed on this machine (Quadro P620 4 GB / i7-9750H)

Hebrew model, 2.5-minute AMR clip, model loading included:

| Device | Speed |
|--------|-------|
| GPU (`int8`) | ~1.6× realtime — a 10-min call ≈ 6 min |
| CPU (`int8`) | ~0.7× realtime — a 10-min call ≈ 14 min |

---

## Measuring accuracy

Don't trust any transcription tool (this one included) on faith. To measure it:

1. Pick a few real calls — a clear one and a noisy one per language.
2. Transcribe them, then fix one copy of each `.txt` by hand → that's the reference.
3. Score machine output against it:

```powershell
phone-transcriber evaluate reference.txt transcripts\call.txt other_run\call.txt
```

It prints **WER** (word error rate) and **CER** (character error rate) — lower
is better. Punctuation, case, niqqud and `ё`/`е` are ignored. Numbers are not
normalized: "340" vs "שלוש מאות וארבעים" counts as an error, so write numbers
in the reference the way the model does (digits).

Use this to compare models (`--model`) or another service on the same calls.

---

## Project structure

```
phone-transcriber/
├── pyproject.toml
├── README.md
└── phone_transcriber/
    ├── cli.py           ← commands: transcribe, evaluate
    ├── models.py        ← language → model mapping
    ├── transcriber.py   ← faster-whisper wrapper, GPU/CPU selection
    ├── output.py        ← .txt / .srt writers
    └── evaluate.py      ← WER / CER scoring
```

## Troubleshooting

**Hebrew looks garbled / reversed in the terminal** — terminal rendering only.
Open the `.txt` in Notepad or VS Code.

**`[model] GPU load failed … falling back to CPU`** — the CUDA DLLs are
missing (reinstall with `".[gpu]"`) or the GPU is out of memory (close other
GPU programs). The run continues on the CPU.

**`open() got an unexpected keyword argument 'metadata_errors'`** — PyAV ≥ 19
is installed; `pyproject.toml` pins `av<19`. Reinstall with `pip install -e ".[gpu]"`.
