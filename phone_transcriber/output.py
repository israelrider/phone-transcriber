"""Writing transcripts to disk."""

from __future__ import annotations

from pathlib import Path

from phone_transcriber.transcriber import Transcript


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_text(transcript: Transcript) -> str:
    return "\n".join(s.text for s in transcript.segments if s.text) + "\n"


def to_srt(transcript: Transcript) -> str:
    blocks = []
    for i, s in enumerate((s for s in transcript.segments if s.text), start=1):
        blocks.append(f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{s.text}\n")
    return "\n".join(blocks)


def write(transcript: Transcript, txt_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(to_text(transcript), encoding="utf-8")
    txt_path.with_suffix(".srt").write_text(to_srt(transcript), encoding="utf-8")
