"""Which Whisper model to use for which language."""

from __future__ import annotations

# ivrit.ai's Hebrew fine-tune of large-v3-turbo, in CTranslate2 format.
HEBREW_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"

# Stock multilingual model; strong on Russian and English.
GENERAL_MODEL = "large-v3-turbo"

MODEL_FOR_LANGUAGE = {
    "he": HEBREW_MODEL,
    "ru": GENERAL_MODEL,
    "en": GENERAL_MODEL,
}

LANGUAGES = tuple(MODEL_FOR_LANGUAGE)

# Most calls are Hebrew; auto-detection falls back to this when unsure.
DEFAULT_LANGUAGE = "he"


def model_for(language: str) -> str:
    return MODEL_FOR_LANGUAGE.get(language, GENERAL_MODEL)
