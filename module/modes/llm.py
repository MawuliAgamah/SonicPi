"""Camera or image snapshot to ElevenLabs music."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from main import run_image_to_music
from module.camera import capture_frame


def run_camera_to_music(
    *,
    camera_index: int = 0,
    camera_backend: str = "auto",
    warmup_seconds: float = 1.0,
    snapshot_dir: str | Path = "captured_images",
    output_dir: str | Path = "generated_music",
    save_extracted_parts: bool = False,
    max_description_parts: int = 6,
    music_model: str | None = None,
    output_format: str | None = None,
    music_length_ms: int | None = 30000,
    instrumental_only: bool = False,
) -> dict[str, object]:
    """Capture one camera frame, then run the existing image-to-music flow."""

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for camera LLM mode") from exc

    captured = capture_frame(
        camera_index=camera_index,
        backend=camera_backend,  # type: ignore[arg-type]
        warmup_seconds=warmup_seconds,
    )
    snapshot_path = Path(snapshot_dir)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    image_path = snapshot_path / f"camera_snapshot_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    ok = cv2.imwrite(str(image_path), captured.frame)
    if not ok:
        raise OSError(f"Could not write camera snapshot: {image_path}")

    summary = run_image_to_music(
        image_path,
        output_dir=output_dir,
        save_extracted_parts=save_extracted_parts,
        max_description_parts=max_description_parts,
        music_model=music_model,
        output_format=output_format,
        music_length_ms=music_length_ms,
        instrumental_only=instrumental_only,
    )
    summary["camera_backend"] = captured.backend
    summary["snapshot"] = str(image_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate music from a camera snapshot or image")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Optional image path for offline testing")
    source.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index for camera capture")
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "opencv", "picamera2"),
        default="auto",
        help="Camera backend; auto prefers PiCamera2 then OpenCV",
    )
    parser.add_argument("--camera-warmup-seconds", type=float, default=1.0, help="Camera warmup before snapshot")
    parser.add_argument("--snapshot-dir", default="captured_images", help="Where camera snapshots are saved")
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
    if args.image:
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
    else:
        summary = run_camera_to_music(
            camera_index=args.camera_index,
            camera_backend=args.camera_backend,
            warmup_seconds=args.camera_warmup_seconds,
            snapshot_dir=args.snapshot_dir,
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

    if summary.get("snapshot"):
        print(f"Camera snapshot: {summary['snapshot']}")
        print(f"Camera backend: {summary.get('camera_backend')}")
    print("Description:")
    print(summary["description"])
    print("\nElevenLabs prompt:")
    print(summary["prompt"])
    print("\nSaved files:")
    for label, path in summary["saved"].items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
