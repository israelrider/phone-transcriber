from __future__ import annotations

from pathlib import Path


def transcribe(audio_path: str | Path, whisper_model: str = "medium") -> str:
    """Transcribe *audio_path* to text using a local Whisper model.

    Downloads the model on first run (~1.5 GB for 'medium').
    Language is forced to Hebrew so Whisper skips auto-detection.
    """
    try:
        import whisper  # openai-whisper
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        ) from exc

    print(f"[whisper] loading model '{whisper_model}' …")
    model = whisper.load_model(whisper_model)

    print(f"[whisper] transcribing '{audio_path}' …")
    result = model.transcribe(
        str(audio_path),
        language="he",
        task="transcribe",
        verbose=False,
    )

    text: str = result["text"]  # type: ignore[index]
    return text.strip()
