"""Audio-to-text with faster-whisper (CTranslate2)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _add_cuda_dll_dirs() -> None:
    """Make the pip-installed nvidia-* DLLs (cuBLAS, cuDNN) loadable on Windows."""
    if sys.platform != "win32":
        return
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for bin_dir in nvidia.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


_add_cuda_dll_dirs()
# Without Developer Mode the HF cache copies files instead of symlinking; that's fine.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import ctranslate2  # noqa: E402
from faster_whisper import WhisperModel, decode_audio  # noqa: E402

from phone_transcriber.models import GENERAL_MODEL, model_for  # noqa: E402


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str
    language_probability: float | None
    model: str
    duration: float
    segments: list[Segment]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


class Transcriber:
    """Loads models lazily and keeps them for the rest of the batch."""

    def __init__(self, device: str = "auto", compute_type: str = "int8") -> None:
        self.device = resolve_device(device)
        self.compute_type = compute_type
        self._models: dict[str, WhisperModel] = {}

    def _load(self, name: str) -> WhisperModel:
        if name not in self._models:
            print(f"[model] loading {name} on {self.device} ({self.compute_type}) …")
            try:
                self._models[name] = WhisperModel(
                    name, device=self.device, compute_type=self.compute_type
                )
            except RuntimeError as exc:
                if self.device != "cuda":
                    raise
                # Out of VRAM or missing CUDA libraries: CPU is slower but works.
                print(f"[model] GPU load failed ({exc}); falling back to CPU")
                self.device = "cpu"
                self._models.clear()
                return self._load(name)
        return self._models[name]

    def detect_language(self, audio) -> tuple[str, float]:
        model = self._load(GENERAL_MODEL)
        language, probability, _ = model.detect_language(
            audio, vad_filter=True, language_detection_segments=3
        )
        return language, probability

    def transcribe(
        self, path: Path, language: str = "he", model_name: str | None = None
    ) -> Transcript:
        audio = decode_audio(str(path))
        probability = None
        if language == "auto":
            language, probability = self.detect_language(audio)
            print(f"[lang] detected '{language}' (p={probability:.2f})")

        name = model_name or model_for(language)
        model = self._load(name)
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            # Stops one misheard phrase from snowballing into a repetition loop,
            # a common failure on noisy phone audio.
            condition_on_previous_text=False,
        )
        result = [Segment(s.start, s.end, s.text.strip()) for s in segments]
        return Transcript(language, probability, name, info.duration, result)
