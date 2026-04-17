"""Root orchestration entrypoint for image-to-music generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from module import (
    build_elevenlabs_prompt,
    describe_image,
    extract_features,
    extract_image_parts,
    generate_elevenlabs_music,
    load_image,
    save_elevenlabs_music_result,
    save_parts,
)


def run_image_to_music(
    image_path: str | Path,
    *,
    output_dir: str | Path = "generated_music",
    save_extracted_parts: bool = False,
    max_description_parts: int = 6,
    music_model: str | None = None,
    output_format: str | None = None,
    music_length_ms: int | None = 30000,
    instrumental_only: bool = False,
) -> dict[str, object]:
    """Describe an image, build a music prompt, call ElevenLabs, and save results."""

    image = load_image(image_path)
    features, _, _ = extract_features(image)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_extracted_parts:
        save_parts(extract_image_parts(image), output_path / "image_parts")

    description = describe_image(
        image,
        include_parts=True,
        max_parts=max_description_parts,
    )
    prompt = build_elevenlabs_prompt(
        description.description,
        features=description.features or features,
        instrumental_only=instrumental_only,
    )
    music = generate_elevenlabs_music(
        prompt,
        model_id=music_model,
        output_format=output_format,
        music_length_ms=music_length_ms,
        force_instrumental=instrumental_only,
    )

    stem = f"image_to_music_{time.strftime('%Y%m%d-%H%M%S')}"
    saved = save_elevenlabs_music_result(music, output_path, stem=stem)

    summary = {
        "image": str(image_path),
        "description": description.description,
        "prompt": prompt,
        "features": description.features,
        "music_provider": "elevenlabs",
        "music_model": music.model_id,
        "output_format": music.output_format,
        "song_id": music.song_id,
        "provider_metadata": music.metadata,
        "saved": {key: str(path) for key, path in saved.items()},
    }
    summary_path = output_path / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["saved"]["summary"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate music from an image")
    parser.add_argument("image", help="Path to the image to describe")
    parser.add_argument("--output-dir", default="generated_music", help="Directory for generated files")
    parser.add_argument("--save-parts", action="store_true", help="Save OpenCV image crops")
    parser.add_argument("--max-description-parts", type=int, default=6, help="Maximum crops to send to OpenAI")
    parser.add_argument("--music-model", help="ElevenLabs model ID, defaults to ELEVENLABS_MUSIC_MODEL or music_v1")
    parser.add_argument("--output-format", help="ElevenLabs output format, e.g. mp3_44100_128")
    parser.add_argument("--music-length-ms", type=int, default=30000, help="Generated music length in milliseconds")
    parser.add_argument("--instrumental-only", action="store_true", help="Ask ElevenLabs for no vocals")
    parser.add_argument("--json", action="store_true", help="Print full JSON summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_image_to_music(
        args.image,
        output_dir=args.output_dir,
        save_extracted_parts=args.save_parts,
        max_description_parts=args.max_description_parts,
        music_model=args.music_model,
        output_format=args.output_format,
        music_length_ms=args.music_length_ms,
        instrumental_only=args.instrumental_only,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("Description:")
    print(summary["description"])
    print("\nElevenLabs prompt:")
    print(summary["prompt"])
    print("\nSaved files:")
    for label, path in summary["saved"].items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
