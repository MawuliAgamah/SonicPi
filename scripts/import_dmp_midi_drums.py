"""Import gvellut/dmp_midi JSON drum patterns as General MIDI drum files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


REPO_ARCHIVE_URL = "https://github.com/gvellut/dmp_midi/archive/refs/heads/master.zip"
DRUM_CHANNEL = 9  # General MIDI channel 10, zero-based in MIDI files.
TICKS_PER_BEAT = 480
DEFAULT_TEMPO_BPM = 110

GM_DRUM_MAP = {
    "BassDrum": 36,
    "SnareDrum": 38,
    "RimShot": 37,
    "Clap": 39,
    "ClosedHiHat": 42,
    "OpenHiHat": 46,
    "LowTom": 45,
    "MediumTom": 47,
    "HighTom": 50,
    "Cymbal": 51,
    "Cowbell": 56,
    "Tambourine": 54,
}


def import_dmp_midi_drums(
    output_dir: str | Path,
    *,
    archive_url: str = REPO_ARCHIVE_URL,
    repeats: int = 8,
    tempo_bpm: int = DEFAULT_TEMPO_BPM,
    keep_existing: bool = False,
) -> dict[str, object]:
    try:
        import mido
    except ModuleNotFoundError as exc:
        raise RuntimeError("mido is required. Run: pip install -r requirements.txt") from exc

    output_path = Path(output_dir).expanduser()
    if output_path.exists() and not keep_existing:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    patterns = _download_patterns(archive_url)
    tracks = []
    seen_names: set[str] = set()

    for source_name, pattern in patterns:
        title = str(pattern.get("title") or "drum_pattern")
        safe_title = _safe_name(title)
        filename = _unique_filename(safe_title, seen_names)
        midi_path = output_path / f"{filename}.mid"
        _write_drum_midi(
            pattern,
            midi_path,
            mido=mido,
            repeats=repeats,
            tempo_bpm=tempo_bpm,
        )
        tracks.append(_manifest_entry(pattern, midi_path, output_path, source_name, repeats, tempo_bpm))

    manifest = {
        "version": 1,
        "source": {
            "name": "gvellut/dmp_midi",
            "url": "https://github.com/gvellut/dmp_midi/tree/master/input",
            "archive_url": archive_url,
            "license_note": "See upstream repository for licensing details.",
        },
        "drum_mapping": "General MIDI percussion on channel 10",
        "tracks": tracks,
    }
    manifest_path = output_path / "midi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_path),
        "manifest": str(manifest_path),
        "patterns": len(tracks),
        "repeats": repeats,
        "tempo_bpm": tempo_bpm,
    }


def _download_patterns(archive_url: str) -> list[tuple[str, dict[str, Any]]]:
    patterns: list[tuple[str, dict[str, Any]]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "dmp_midi.zip"
        print(f"Downloading {archive_url}", flush=True)
        urllib.request.urlretrieve(archive_url, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            json_names = sorted(
                name
                for name in archive.namelist()
                if "/input/" in name and name.endswith(".json")
            )
            for name in json_names:
                data = json.loads(archive.read(name).decode("utf-8"))
                if not isinstance(data, list):
                    raise ValueError(f"Expected a JSON list in {name}")
                for pattern in data:
                    if not isinstance(pattern, dict):
                        continue
                    patterns.append((Path(name).name, pattern))
    return patterns


def _write_drum_midi(
    pattern: dict[str, Any],
    midi_path: Path,
    *,
    mido,
    repeats: int,
    tempo_bpm: int,
) -> None:
    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("track_name", name=str(pattern.get("title") or midi_path.stem), time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    numerator, denominator = _parse_signature(str(pattern.get("signature") or "4/4"))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=numerator,
            denominator=denominator,
            time=0,
        )
    )

    length = int(pattern.get("length") or 16)
    step_ticks = _step_ticks(numerator, denominator, length)
    events: dict[int, list[tuple[str, int, int]]] = {}
    accent = pattern.get("accent") or []

    for repeat in range(repeats):
        offset = repeat * length * step_ticks
        for instrument, steps in (pattern.get("tracks") or {}).items():
            note = GM_DRUM_MAP.get(instrument)
            if note is None:
                continue
            for index, value in enumerate(steps):
                if value != "Note":
                    continue
                start = offset + index * step_ticks
                velocity = 118 if index < len(accent) and accent[index] == "Accent" else 92
                duration = max(30, int(step_ticks * 0.6))
                events.setdefault(start, []).append(("on", note, velocity))
                events.setdefault(start + duration, []).append(("off", note, 0))

    current_tick = 0
    for tick in sorted(events):
        delta = tick - current_tick
        current_tick = tick
        for event_index, (_, note, velocity) in enumerate(events[tick]):
            time_delta = delta if event_index == 0 else 0
            if velocity:
                track.append(mido.Message("note_on", channel=DRUM_CHANNEL, note=note, velocity=velocity, time=time_delta))
            else:
                track.append(mido.Message("note_off", channel=DRUM_CHANNEL, note=note, velocity=0, time=time_delta))

    track.append(mido.MetaMessage("end_of_track", time=step_ticks))
    midi.save(midi_path)


def _manifest_entry(
    pattern: dict[str, Any],
    midi_path: Path,
    root: Path,
    source_name: str,
    repeats: int,
    tempo_bpm: int,
) -> dict[str, Any]:
    title = str(pattern.get("title") or midi_path.stem)
    instruments = sorted((pattern.get("tracks") or {}).keys())
    style = _style_from_title(title)
    mood = _mood_from_style(style, title)
    length = int(pattern.get("length") or 16)
    numerator, denominator = _parse_signature(str(pattern.get("signature") or "4/4"))
    beats_per_pattern = numerator * 4 / denominator
    duration_seconds = round((beats_per_pattern * repeats) * (60 / tempo_bpm), 2)
    return {
        "file": midi_path.relative_to(root).as_posix(),
        "title": title,
        "composer": "gvellut/dmp_midi",
        "role": "drums",
        "style": style,
        "mood": mood,
        "energy": _energy_from_title(title),
        "tempo_bpm": tempo_bpm,
        "duration_seconds": duration_seconds,
        "loopable": True,
        "signature": pattern.get("signature") or "4/4",
        "steps": length,
        "repeats": repeats,
        "drum_mapping": {name: GM_DRUM_MAP[name] for name in instruments if name in GM_DRUM_MAP},
        "tags": sorted(set([style, "drums", "general midi", source_name.removesuffix(".json").lower()] + [_split_title(title)] + instruments)),
        "notes": f"Converted from {source_name}. General MIDI drums on channel 10.",
    }


def _parse_signature(signature: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\s*/\s*(\d+)$", signature)
    if not match:
        return 4, 4
    return int(match.group(1)), int(match.group(2))


def _step_ticks(numerator: int, denominator: int, length: int) -> int:
    quarter_notes = numerator * 4 / denominator
    return max(1, int(TICKS_PER_BEAT * quarter_notes / length))


def _style_from_title(title: str) -> str:
    compact = title.lower()
    if "rock" in compact:
        return "rock drums"
    if "funk" in compact:
        return "funk drums"
    if "disco" in compact:
        return "disco drums"
    if "reggae" in compact:
        return "reggae drums"
    if "latin" in compact or "afro" in compact or "bossa" in compact or "samba" in compact:
        return "latin drums"
    if "blues" in compact:
        return "blues drums"
    if "jazz" in compact or "swing" in compact:
        return "jazz drums"
    if "break" in compact:
        return "drum break"
    return "drum pattern"


def _mood_from_style(style: str, title: str) -> str:
    if "funk" in style or "disco" in style:
        return "groovy, bright, danceable"
    if "rock" in style:
        return "driving, direct, energetic"
    if "reggae" in style:
        return "relaxed, warm, offbeat"
    if "latin" in style:
        return "syncopated, colorful, lively"
    if "jazz" in style or "blues" in style:
        return "loose, expressive, human"
    if "break" in title.lower():
        return "transitional, energetic, accented"
    return "rhythmic, adaptable, steady"


def _energy_from_title(title: str) -> str:
    compact = title.lower()
    if "break" in compact or "rock" in compact or "disco" in compact:
        return "high"
    if "reggae" in compact or "blues" in compact:
        return "medium"
    return "medium"


def _split_title(title: str) -> str:
    return " ".join(part.lower() for part in re.findall(r"[A-Z]?[a-z]+|[0-9]+", title))


def _safe_name(title: str) -> str:
    split = _split_title(title)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", split).strip("_")
    return safe or "drum_pattern"


def _unique_filename(base: str, seen_names: set[str]) -> str:
    candidate = base
    counter = 2
    while candidate in seen_names:
        candidate = f"{base}_{counter}"
        counter += 1
    seen_names.add(candidate)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import dmp_midi JSON drum patterns as MIDI")
    parser.add_argument("--output-dir", default="midi/dmp_drum_patterns", help="Destination folder")
    parser.add_argument("--archive-url", default=REPO_ARCHIVE_URL, help="GitHub archive URL")
    parser.add_argument("--repeats", type=int, default=8, help="How many times to repeat each pattern")
    parser.add_argument("--tempo-bpm", type=int, default=DEFAULT_TEMPO_BPM, help="Tempo for generated MIDI files")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear output dir before importing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = import_dmp_midi_drums(
        args.output_dir,
        archive_url=args.archive_url,
        repeats=args.repeats,
        tempo_bpm=args.tempo_bpm,
        keep_existing=args.keep_existing,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
