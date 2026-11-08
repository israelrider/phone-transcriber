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
# otherwise garbles ("Хлюми" for Битуах Леуми, "Ём-шли-шли" for יום שלישי).
# Each entry is (hotword, Hebrew, pattern). The Cyrillic hotword is given to
# the model (Hebrew script there turns the output into gibberish); then every
# match of the pattern in the transcript is replaced by the Hebrew spelling.
#
# Patterns match case-insensitively at a word start. None means "the hotword
# plus any Russian case ending". A pattern lists the garbled spellings actually
# seen in transcripts, and must not catch ordinary Russian words ("рана",
# "Варшава", lowercase "партнер"). Order matters: longer phrases first.
Term = tuple[str, str, str | None]

# "йом" (day) as the model writes it: "йом", "Ём", "Ем", "Пьем", "Бьем",
# "объем" (from "в йом"). Only ever matched right before a day name.
_YOM = r"(?:йо|[пб]ь[её]|объ[её]|[её])м[\s-]*"

# Weekdays: (hotword, Hebrew, spellings safe on their own, extra after "йом").
# "рыбий" and "шли шли" are real Russian, so they only count right after "йом".
_DAYS = [
    ("ришон", "ראשון", r"ришон", None),
    ("шени", "שני", r"шени", None),
    ("шлиши", "שלישי", r"шли-?(?:ши|шли)|шлиф(?:ы|ши)", r"шли\s+шли"),
    ("ревии", "רביעי", r"ре?вии", r"рыб(?:ий|ьи)(?!\s+жир)"),
    ("хамиши", "חמישי", r"х[аоу]м[еияю]ши", None),
    ("шиши", "שישי", r"шиши", None),
]

# Weekdays are fixed up in the text but NOT sent as hotwords: six similar
# names in the prompt made the model hear "йом шиши" (Friday) where it had
# correctly heard "Ём-шли-шли" (Tuesday) without them.
_DAY_TERMS: list[Term] = [
    *[
        (f"йом {ru}", f"יום {he}", rf"{_YOM}(?:{alone}|{after_yom or alone})\b")
        for ru, he, alone, after_yom in _DAYS
    ],
    *[(ru, he, rf"(?:{alone})\b") for ru, he, alone, _ in _DAYS],
]

_HOTWORD_TERMS: list[Term] = [
    ("Битуах Леуми", "ביטוח לאומי", None),
    ("Рами Леви", "רמי לוי", None),
    ("Селком", "סלקום", None),
    # Case-sensitive: lowercase "партнер" is the Russian word.
    ("Партнер", "פרטנר", r"(?-i:Партн[её]р)\w*"),
    ("Пелефон", "פלאפון", None),
    ("Голан Телеком", "גולן טלקום", None),
    ("Голан", "גולן", None),
    ("Безек", "בזק", None),
    ("купат холим", "קופת חולים", None),
    ("Маккаби", "מכבי", None),
    ("Клалит", "כללית", None),
    ("тикшорет", "תקשורת", None),
    ("шерут", "שירות", None),
    ("ишур", "אישור", None),
    ("мамад", 'ממ"ד', None),
    ("Раанана", "רעננה", r"раа?нан\w*"),
    ("Кфар-Саба", "כפר סבא", r"кфар[\s-]*саб\w*|варса[вб]\w*"),
    ("мошавы", "מושבים", r"(?:мош|марш)ав(?:ы|ов|ах|ам|ами)\b"),
    ("мошав", "מושב", r"(?:мош|марш)ав\w*"),
]

RU_HEBREW_TERMS: list[Term] = _DAY_TERMS + _HOTWORD_TERMS

HOTWORDS = {"ru": ", ".join(dict.fromkeys(ru for ru, _, _ in _HOTWORD_TERMS))}

# Most calls are Hebrew; auto-detection falls back to this when unsure.
DEFAULT_LANGUAGE = "he"


def model_for(language: str) -> str:
    return MODEL_FOR_LANGUAGE.get(language, GENERAL_MODEL)
