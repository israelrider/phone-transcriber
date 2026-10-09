from __future__ import annotations

import argparse
import time
from pathlib import Path

from phone_transcriber.models import LANGUAGES

AUDIO_SUFFIXES = {
    ".amr", ".mp3", ".m4a", ".wav", ".ogg", ".opus", ".3gp", ".aac", ".flac", ".wma",
}


def _collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            files.extend(
                sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES)
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f"[skip] not found: {p}")
    return files


def _transcribe(args: argparse.Namespace) -> int:
    files = _collect_inputs(args.inputs)
    if not files:
        print("Error: no audio files to transcribe.")
        return 1

    # Imported here so `--help` and `evaluate` don't pay for loading CUDA.
    from phone_transcriber import output
    from phone_transcriber.transcriber import Transcriber

    out_dir = Path(args.out_dir).expanduser()
    engine = Transcriber(device=args.device, compute_type=args.compute_type)
    failures = 0

    for i, path in enumerate(files, start=1):
        txt_path = out_dir / f"{path.stem}.txt"
        if txt_path.exists() and not args.force:
            print(f"[{i}/{len(files)}] exists, skipping (use --force): {txt_path}")
            continue
        print(f"[{i}/{len(files)}] {path.name}")
        started = time.perf_counter()
        try:
            transcript = engine.transcribe(path, language=args.lang, model_name=args.model)
        except Exception as exc:  # keep going through the rest of a batch
            failures += 1
            print(f"[error] {path.name}: {exc}")
            continue
        output.write(transcript, txt_path)
        elapsed = time.perf_counter() - started
        speed = transcript.duration / elapsed if elapsed else 0
        print(
            f"[done] {transcript.duration:.0f}s of audio in {elapsed:.0f}s "
            f"({speed:.1f}x realtime) -> {txt_path}"
        )

    return 1 if failures else 0


def _evaluate(args: argparse.Namespace) -> int:
    from phone_transcriber.evaluate import report

    report(Path(args.reference), [Path(h) for h in args.hypotheses])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phone-transcriber",
        description="Transcribe phone-call recordings (Hebrew, Russian, English) locally.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transcribe audio files or folders")
    t.add_argument("inputs", nargs="+", help="Audio files and/or folders containing them")
    t.add_argument(
        "--lang",
        # Not "he": the Hebrew model transcribes Russian/English audio as Hebrew-letter
        # gibberish without complaint, so detecting per file is the safe default.
        default="auto",
        choices=[*LANGUAGES, "auto"],
        help="Spoken language (default: auto = detect per file). Forcing one skips detection.",
    )
    t.add_argument(
        "--out-dir",
        default="transcripts",
        help="Where to write <name>.txt and <name>.srt (default: ./transcripts)",
    )
    t.add_argument(
        "--model",
        default=None,
        help="Override the per-language model (e.g. ivrit-ai/whisper-large-v3-ct2)",
    )
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    t.add_argument(
        "--compute-type",
        default="int8",
        help="CTranslate2 compute type (default: int8; fits a 4 GB GPU)",
    )
    t.add_argument("--force", action="store_true", help="Redo files that already have output")
    t.set_defaults(func=_transcribe)

    e = sub.add_parser("evaluate", help="Score transcripts against a corrected reference")
    e.add_argument("reference", help="Hand-corrected transcript (.txt)")
    e.add_argument("hypotheses", nargs="+", help="Machine transcripts to score (.txt)")
    e.set_defaults(func=_evaluate)
    return p


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
