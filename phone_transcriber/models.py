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

# Hebrew words common in Israeli Russian-speakers' calls, which the model
# otherwise garbles ("Хлюми" for Битуах Леуми). The Cyrillic side is given to
# the model as hotwords (Hebrew script there turns the output into gibberish),
# then swapped for the Hebrew side in the transcript. Add terms as needed.
RU_HEBREW_TERMS = {
    "Битуах Леуми": "ביטוח לאומי",
    "Рами Леви": "רמי לוי",
    "Селком": "סלקום",
    "Партнер": "פרטנר",
    "Пелефон": "פלאפון",
    "Голан Телеком": "גולן טלקום",
    "Голан": "גולן",
    "Безек": "בזק",
    "купат холим": "קופת חולים",
    "Маккаби": "מכבי",
    "Клалит": "כללית",
    "тикшорет": "תקשורת",
    "шерут": "שירות",
    "ишур": "אישור",
}

HOTWORDS = {"ru": ", ".join(RU_HEBREW_TERMS)}

# Most calls are Hebrew; auto-detection falls back to this when unsure.
DEFAULT_LANGUAGE = "he"


def model_for(language: str) -> str:
    return MODEL_FOR_LANGUAGE.get(language, GENERAL_MODEL)
