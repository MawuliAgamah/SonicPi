"""Pick a SoundFont from a manifest based on the image-derived vibe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .midi_arranger import VibeProfile


DEFAULT_MANIFEST_NAME = "soundfont_manifest.json"
SOUNDFONT_EXTENSIONS = {".sf2", ".sf3"}
VIBE_BUCKETS = ("bright", "dark", "muted", "neutral")


@dataclass(frozen=True)
class SoundFontEntry:
    path: Path
    name: str
    best_for: str
    tags: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class SoundFontSelection:
    entry: SoundFontEntry
    bucket: str
    reason: str


def vibe_bucket(vibe: VibeProfile) -> str:
    """Same buckets palette_for uses, so the SoundFont matches the patches."""

    if vibe.brightness > 0.6 and vibe.color > 0.5:
        return "bright"
    if vibe.brightness < 0.3:
        return "dark"
    if vibe.color < 0.25:
        return "muted"
    return "neutral"


def load_soundfont_manifest(
    soundfont_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> list[SoundFontEntry]:
    """Load entries whose .sf2 file actually exists, falling back to discovery."""

    root = Path(soundfont_dir).expanduser()
    chosen_manifest = (
        Path(manifest_path).expanduser()
        if manifest_path
        else root / DEFAULT_MANIFEST_NAME
    )

    entries: list[SoundFontEntry] = []
    if chosen_manifest.exists():
        with chosen_manifest.open("r", encoding="utf-8") as file:
            data = json.load(file)
        raw_entries = data.get("soundfonts", []) if isinstance(data, dict) else data
        if not isinstance(raw_entries, list):
            raise ValueError(f"SoundFont manifest must contain a list: {chosen_manifest}")

        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            file_value = str(item.get("file") or "").strip()
            if not file_value:
                continue
            candidate = (root / file_value).expanduser()
            if not candidate.exists():
                continue
            entries.append(_entry_from_manifest_item(item, candidate))

    # Auto-discover anything in the folder that the manifest didn't already list.
    listed = {entry.path.resolve() for entry in entries}
    if root.exists():
        for candidate in sorted(root.glob("*")):
            if candidate.suffix.lower() not in SOUNDFONT_EXTENSIONS:
                continue
            if candidate.resolve() in listed:
                continue
            entries.append(
                SoundFontEntry(
                    path=candidate,
                    name=candidate.stem,
                    best_for="neutral",
                    tags=(),
                    notes="Auto-discovered (not in manifest).",
                )
            )

    return entries


def pick_soundfont(
    vibe: VibeProfile,
    entries: list[SoundFontEntry],
    *,
    fallback: Path | None = None,
) -> SoundFontSelection | None:
    """Pick a SoundFont matching the vibe; fall back to any available entry."""

    if not entries:
        if fallback is None:
            return None
        return SoundFontSelection(
            entry=SoundFontEntry(
                path=fallback,
                name=fallback.stem,
                best_for="neutral",
                tags=(),
                notes="Fallback SoundFont (no manifest entries).",
            ),
            bucket=vibe_bucket(vibe),
            reason="No manifest entries; using the player's default SoundFont.",
        )

    bucket = vibe_bucket(vibe)
    for entry in entries:
        if entry.best_for == bucket:
            return SoundFontSelection(
                entry=entry,
                bucket=bucket,
                reason=f"Matched vibe bucket '{bucket}' via best_for.",
            )

    # Tag fallback — pick the first entry whose tags overlap with the bucket name.
    for entry in entries:
        if bucket in {tag.lower() for tag in entry.tags}:
            return SoundFontSelection(
                entry=entry,
                bucket=bucket,
                reason=f"Matched vibe bucket '{bucket}' via tag overlap.",
            )

    # Final fallback: any "neutral" entry, then the first available.
    for entry in entries:
        if entry.best_for == "neutral":
            return SoundFontSelection(
                entry=entry,
                bucket=bucket,
                reason=f"No '{bucket}' match; using neutral SoundFont.",
            )

    return SoundFontSelection(
        entry=entries[0],
        bucket=bucket,
        reason=f"No '{bucket}' or neutral match; using first available SoundFont.",
    )


def _entry_from_manifest_item(item: dict[str, Any], path: Path) -> SoundFontEntry:
    best_for = str(item.get("best_for") or "neutral").lower()
    if best_for not in VIBE_BUCKETS:
        best_for = "neutral"
    tags_raw = item.get("tags") or []
    tags = tuple(str(tag) for tag in tags_raw if str(tag).strip())
    return SoundFontEntry(
        path=path,
        name=str(item.get("name") or path.stem),
        best_for=best_for,
        tags=tags,
        notes=str(item.get("notes") or ""),
    )
