"""Score transcripts against a hand-corrected reference (WER / CER)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import jiwer

_NIQQUD = re.compile(r"[֑-ׇ]")


def normalize(text: str) -> str:
    """Strip what shouldn't count as an error: case, punctuation, niqqud, spacing."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _NIQQUD.sub("", text).replace("ё", "е")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def score(reference: str, hypothesis: str) -> tuple[float, float]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def report(reference_path: Path, hypothesis_paths: list[Path]) -> None:
    reference = reference_path.read_text(encoding="utf-8")
    print(f"reference: {reference_path}")
    print(f"{'WER':>7} {'CER':>7}  hypothesis")
    for path in hypothesis_paths:
        wer, cer = score(reference, path.read_text(encoding="utf-8"))
        print(f"{wer:7.1%} {cer:7.1%}  {path}")
