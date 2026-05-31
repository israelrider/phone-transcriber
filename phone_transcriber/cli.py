from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from phone_transcriber import ollama_client, transcriber
from phone_transcriber.prompts import CLEANUP_SYSTEM_PROMPT, user_prompt


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output(input_path: Path) -> Path:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    stem = input_path.stem
    return artifacts / f"{stem}_{_stamp()}.txt"


def _run(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return 1

    output_path = (
        Path(args.output).expanduser() if args.output else _default_output(input_path)
    )

    host = ollama_client.get_host()
    model = ollama_client.get_model()

    if args.dry_run:
        print(f"[dry-run] input : {input_path}")
        print(f"[dry-run] output: {output_path}")
        print(f"[dry-run] whisper model  : {args.whisper_model}")
        print(f"[dry-run] ollama host    : {host}")
        print(f"[dry-run] ollama model   : {model}")
        return 0

    # ── Stage 1: local Whisper transcription ─────────────────────────────────
    raw = transcriber.transcribe(input_path, whisper_model=args.whisper_model)
    print(f"[whisper] raw transcript length: {len(raw)} chars")

    # ── Stage 2: Ollama Cloud cleanup ─────────────────────────────────────────
    if args.skip_cleanup:
        final = raw
        print("[ollama] skipping cleanup (--skip-cleanup)")
    else:
        api_key = ollama_client.get_api_key()
        print(f"[ollama] sending to model '{model}' for cleanup …")
        messages = [
            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(raw)},
        ]
        final = ollama_client.chat_completion(
            host=host,
            api_key=api_key,
            model=model,
            messages=messages,
            timeout_s=float(args.timeout_s),
        )

    # ── Write output ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final + "\n", encoding="utf-8")
    print(f"[done] transcript saved: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phone-transcriber",
        description="Transcribe a Hebrew phone-call MP3 to text (Whisper + Ollama Cloud).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("transcribe", help="Transcribe an MP3 file")

    run.add_argument("--input", required=True, help="Path to the MP3 (or WAV/M4A/OGG) file")
    run.add_argument(
        "--output",
        default=None,
        help="Output .txt path (default: artifacts/<stem>_<timestamp>.txt)",
    )
    run.add_argument(
        "--whisper-model",
        default=os.getenv("WHISPER_MODEL", "medium"),
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: medium). Larger = slower but more accurate.",
    )
    run.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip the Ollama Cloud cleanup step; save the raw Whisper output.",
    )
    run.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="HTTP timeout for the Ollama Cloud API call (seconds)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without calling Whisper or the API",
    )
    run.set_defaults(func=_run)
    return p


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True) or None)
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
