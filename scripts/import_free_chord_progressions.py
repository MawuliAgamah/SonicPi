"""Download Free-Chord-Progressions MIDI packs and build a manifest."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ARCHIVE_URL = "https://github.com/BenLeon2001/Free-Chord-Progressions/archive/refs/heads/main.zip"
MIDI_EXTENSIONS = {".mid", ".midi"}
ZIP_EXTENSION = ".zip"
IGNORED_TOKENS = {"mid", "midi", "zip", "ds", "store", "macosx"}


def import_free_chord_progressions(
    output_dir: str | Path,
    *,
    archive_url: str = REPO_ARCHIVE_URL,
    keep_existing: bool = False,
) -> dict[str, object]:
    output_path = Path(output_dir).expanduser()
    if output_path.exists() and not keep_existing:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_tracks: list[dict[str, object]] = []
    extracted_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "free_chord_progressions.zip"
        print(f"Downloading {archive_url}")
        urllib.request.urlretrieve(archive_url, archive_path)

        repo_extract_path = temp_path / "repo"
        repo_extract_path.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(repo_extract_path)

        repo_root = _single_child_dir(repo_extract_path)
        zip_files = sorted(repo_root.rglob(f"*{ZIP_EXTENSION}"))
        loose_midi_files = sorted(
            path for path in repo_root.rglob("*") if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS
        )

        for midi_path in loose_midi_files:
            relative = midi_path.relative_to(repo_root)
            destination = _unique_destination(output_path / relative, output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(midi_path, destination)
            manifest_tracks.append(_metadata_for_file(destination, output_path, source_pack="repo"))
            extracted_count += 1

        for zip_path in zip_files:
            pack_name = _clean_stem(zip_path.stem)
            pack_dir = output_path / _safe_path_name(pack_name)
            pack_dir.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {zip_path.name}")
            with zipfile.ZipFile(zip_path) as pack:
                for member in pack.infolist():
                    if member.is_dir() or Path(member.filename).suffix.lower() not in MIDI_EXTENSIONS:
                        continue
                    member_name = Path(member.filename)
                    destination = _unique_destination(pack_dir / member_name, output_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with pack.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    manifest_tracks.append(_metadata_for_file(destination, output_path, source_pack=pack_name))
                    extracted_count += 1

    manifest = {
        "version": 1,
        "source": {
            "name": "BenLeon2001/Free-Chord-Progressions",
            "url": "https://github.com/BenLeon2001/Free-Chord-Progressions",
            "archive_url": archive_url,
            "license_note": "Repository README describes the progressions as royalty free with no license required.",
        },
        "tracks": sorted(manifest_tracks, key=lambda track: str(track["file"]).lower()),
    }
    manifest_path = output_path / "midi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _remove_ds_store_files(output_path)

    return {
        "output_dir": str(output_path),
        "manifest": str(manifest_path),
        "midi_files": extracted_count,
    }


def _single_child_dir(path: Path) -> Path:
    children = [child for child in path.iterdir() if child.is_dir()]
    if len(children) == 1:
        return children[0]
    return path


def _metadata_for_file(path: Path, root: Path, *, source_pack: str) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    title = _title_from_stem(path.stem)
    tokens = _tokens_from_path(relative, source_pack)
    style = _style_from_tokens(tokens)
    mood = _mood_from_tokens(tokens)
    energy = _energy_from_tokens(tokens)
    tags = sorted(set(tokens + [style, mood, energy, source_pack.lower()]))
    tags = [tag for tag in tags if tag and tag != "unknown"]
    return {
        "file": relative,
        "title": title,
        "composer": "BenLeon2001 / Free-Chord-Progressions",
        "style": style,
        "mood": mood,
        "energy": energy,
        "tags": tags,
        "notes": f"Imported from {source_pack}. Use for scenes matching {mood} / {style}.",
    }


def _tokens_from_path(relative_path: str, source_pack: str) -> list[str]:
    text = f"{source_pack} {relative_path}"
    parts = re.split(r"[^A-Za-z0-9#]+", text.lower())
    return [part for part in parts if part and len(part) > 1 and part not in IGNORED_TOKENS]


def _style_from_tokens(tokens: list[str]) -> str:
    token_set = set(tokens)
    if "edm" in token_set:
        return "edm"
    if "hip" in token_set or "hop" in token_set or "hiphop" in token_set:
        return "hip hop"
    if "reggae" in token_set:
        return "reggae"
    if "jazz" in token_set or "dominant" in token_set:
        return "jazz harmony"
    if "locrian" in token_set or "phrygian" in token_set or "dorian" in token_set:
        return "modal harmony"
    if "minor" in token_set:
        return "minor progression"
    if "major" in token_set:
        return "major progression"
    return "chord progression"


def _mood_from_tokens(tokens: list[str]) -> str:
    token_set = set(tokens)
    if token_set & {"minor", "locrian", "diminished", "dark", "sad"}:
        return "dark, tense, introspective"
    if token_set & {"major", "happy", "bright"}:
        return "bright, open, uplifting"
    if token_set & {"edm", "dance", "house", "trance"}:
        return "energetic, colorful, driving"
    if token_set & {"reggae"}:
        return "warm, relaxed, sunny"
    if token_set & {"hip", "hop", "trap"}:
        return "cool, rhythmic, urban"
    if token_set & {"dominant", "jazz"}:
        return "sophisticated, unresolved, jazzy"
    return "neutral, harmonic, adaptable"


def _energy_from_tokens(tokens: list[str]) -> str:
    token_set = set(tokens)
    if token_set & {"edm", "dance", "house", "trance", "fast"}:
        return "high"
    if token_set & {"ambient", "slow", "sad", "minor", "dark"}:
        return "low"
    return "medium"


def _title_from_stem(stem: str) -> str:
    return re.sub(r"[_-]+", " ", stem).strip() or stem


def _clean_stem(stem: str) -> str:
    return re.sub(r"\s+", " ", stem).strip()


def _safe_path_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "", name).strip()
    return safe.replace(" ", "_") or "midi_pack"


def _unique_destination(path: Path, root: Path) -> Path:
    destination = path
    counter = 2
    while destination.exists():
        destination = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1

    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside output directory: {destination}") from exc
    return destination


def _remove_ds_store_files(root: Path) -> None:
    for path in root.rglob(".DS_Store"):
        path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import MIDI files from Free-Chord-Progressions")
    parser.add_argument(
        "--output-dir",
        default="midi/free_chord_progressions",
        help="Destination folder for extracted MIDI files and midi_manifest.json",
    )
    parser.add_argument("--archive-url", default=REPO_ARCHIVE_URL, help="GitHub archive URL to download")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear output dir before importing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = import_free_chord_progressions(
        args.output_dir,
        archive_url=args.archive_url,
        keep_existing=args.keep_existing,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
