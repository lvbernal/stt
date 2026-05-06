from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import whisper

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe all audio files in a folder with OpenAI Whisper."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the input folder with audio files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the output folder where .txt files will be written.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help=(
            "Whisper model name to use (tiny, base, small, medium, large, etc.). "
            "Default: medium"
        ),
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Language code (for example: en, es). If omitted, Whisper auto-detects.",
    )
    parser.add_argument(
        "--device",
        help="Torch device to run on (for example: cpu, cuda).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search width for decoding. Higher is slower but usually more stable.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidates when sampling. Default: 5",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Condition each segment on previous text. "
            "Default: disabled to reduce repetition loops."
        ),
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable fp16. Example: --fp16 or --no-fp16",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subfolders recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output .txt files.",
    )
    parser.add_argument(
        "--single-line",
        action="store_true",
        help="Write output as a single line instead of one line per Whisper segment.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Prefix each transcribed line with [hh:mm:ss -> hh:mm:ss].",
    )
    return parser


def get_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_dir}")

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    audio_files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(audio_files)


def transcribe_audio(
    model: whisper.Whisper, audio_path: Path, args: argparse.Namespace
) -> str:
    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    options: dict[str, object] = {
        "task": "transcribe",
        "language": args.language,
        "temperature": args.temperature,
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "condition_on_previous_text": args.condition_on_previous_text,
    }

    if args.fp16 is not None:
        options["fp16"] = args.fp16

    cleaned_options = {k: v for k, v in options.items() if v is not None}
    result = model.transcribe(str(audio_path), verbose=True, **cleaned_options)

    return format_output_text(result, single_line=args.single_line, timestamps=args.timestamps)


def format_output_text(
    result: dict[str, Any], single_line: bool, timestamps: bool
) -> str:
    if single_line:
        return str(result.get("text", "")).strip()

    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list):
        return str(result.get("text", "")).strip()

    lines: list[str] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        if timestamps:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            lines.append(f"[{fmt_seconds(start)} -> {fmt_seconds(end)}] {text}")
        else:
            lines.append(text)

    return "\n".join(lines).strip() or str(result.get("text", "")).strip()


def fmt_seconds(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        audio_files = get_audio_files(args.input_dir, args.recursive)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not audio_files:
        print("No audio files found in the input folder.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = whisper.load_model(args.model, device=args.device)
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        return 1

    completed = 0
    failed = 0
    skipped = 0

    for audio_path in audio_files:
        rel_path = audio_path.relative_to(args.input_dir).with_suffix(".txt")
        output_path = args.output_dir / rel_path

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {audio_path} (output exists)")
            skipped += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            text = transcribe_audio(model, audio_path, args)
            output_path.write_text(text + "\n", encoding="utf-8")
            print(f"[OK] {audio_path} -> {output_path}")
            completed += 1
        except Exception as exc:
            print(f"[FAIL] {audio_path}: {exc}", file=sys.stderr)
            failed += 1

    print(
        f"Done. success={completed} skipped={skipped} failed={failed} total={len(audio_files)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
