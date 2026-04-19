"""Unified launcher for the project modes.

This is intentionally a thin dispatcher around the current entrypoints. It gives
the Pi one command to run while the larger refactor moves shared code into
module/modes over time.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


MODES = {
    "ambient": ("ambient", "Continuous camera-driven local ambient synthesis"),
    "llm": ("module.modes.llm", "Camera or image to OpenAI description to ElevenLabs music"),
    "arrangement": ("mac", "Camera snapshot to MIDI arrangement and playback"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one VideoToAudio mode",
        usage=(
            "python3 app.py {ambient,llm,arrangement} [mode options]\n\n"
            "Examples:\n"
            "  python3 app.py ambient --host 0.0.0.0 --port 8091\n"
            "  python3 app.py llm --camera-index 0 --output-dir generated_music\n"
            "  python3 app.py llm --image path/to/image.jpg --output-dir generated_music\n"
            "  python3 app.py arrangement --host 0.0.0.0 --port 8090"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=sorted(MODES), help="Which pipeline to run")
    parser.add_argument(
        "mode_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected mode",
    )
    return parser


def dispatch(mode: str, mode_args: list[str]) -> None:
    module_name, _ = MODES[mode]
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Mode module {module_name!r} does not expose main()")

    original_argv = sys.argv[:]
    try:
        sys.argv = [f"{Path(sys.argv[0]).name} {mode}", *mode_args]
        module.main()
    finally:
        sys.argv = original_argv


def main() -> None:
    args = build_parser().parse_args()
    dispatch(args.mode, args.mode_args)


if __name__ == "__main__":
    main()
