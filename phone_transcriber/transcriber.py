"""Audio-to-text with faster-whisper (CTranslate2)."""

from __future__ import annotations

import os
import re
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

from phone_transcriber.models import (  # noqa: E402
    DEFAULT_LANGUAGE,
    GENERAL_MODEL,
    HOTWORDS,
    LANGUAGES,
    RU_HEBREW_TERMS,
    model_for,
)


def hebraize(text: str) -> str:
    """Write Hebrew terms in Hebrew script, dropping Russian case endings."""
    for ru, he in RU_HEBREW_TERMS.items():
        text = re.sub(rf"\b{ru}\w*", he, text)
    return text

DETECTION_WINDOWS = 4
# Share of (he + ru + en) a non-default language needs before we switch to it.
CONFIDENT_SHARE = 0.7


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
        """Pick one of LANGUAGES, assuming Hebrew unless another is clearly likelier.

        On narrowband phone audio Whisper's detector is close to a coin flip
        (a real Hebrew call scored he 27% / en 23% / ru 18%), and guessing wrong
        makes the general model *translate* the call into invented English.
        So: average the scores over several 30 s windows, compare only our
        languages, and leave Hebrew only on a confident result.
        """
        model = self._load(GENERAL_MODEL)
        window = 30 * 16000  # decode_audio() returns 16 kHz samples
        starts = range(0, max(len(audio) - window // 3, 1), window)
        totals = dict.fromkeys(LANGUAGES, 0.0)
        for start in list(starts)[:DETECTION_WINDOWS]:
            _, _, scores = model.detect_language(
                audio[start : start + window], vad_filter=True
            )
            for lang, p in scores:
                if lang in totals:
                    totals[lang] += p

        total = sum(totals.values()) or 1.0
        shares = {lang: p / total for lang, p in totals.items()}
        best = max(shares, key=shares.get)
        summary = " / ".join(f"{lang} {shares[lang]:.0%}" for lang in LANGUAGES)
        if best != DEFAULT_LANGUAGE and shares[best] < CONFIDENT_SHARE:
            print(
                f"[lang] WARNING: uncertain ({summary}) -> using '{DEFAULT_LANGUAGE}'. "
                f"If wrong, rerun with --lang {best} --force"
            )
            best = DEFAULT_LANGUAGE
        else:
            print(f"[lang] {summary}")
        return best, shares[best]

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
            # Unlike initial_prompt, hotwords reach every 30 s window.
            hotwords=HOTWORDS.get(language),
        )
        result = [Segment(s.start, s.end, hebraize(s.text.strip())) for s in segments]
        return Transcript(language, probability, name, info.duration, result)
