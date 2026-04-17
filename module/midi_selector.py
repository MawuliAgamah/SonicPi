"""Choose a local MIDI file based on an image description."""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .image_describer import describe_image


DEFAULT_OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
MIDI_EXTENSIONS = {".mid", ".midi"}
DEFAULT_MANIFEST_NAME = "midi_manifest.json"


@dataclass(frozen=True)
class MidiCandidate:
    """A playable local MIDI file."""

    index: int
    path: Path
    title: str = ""
    composer: str = ""
    style: str = ""
    mood: str = ""
    energy: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        return self.title or _filename_to_label(self.path.stem)

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "filename": self.filename,
            "label": self.label,
            "composer": self.composer,
            "style": self.style,
            "mood": self.mood,
            "energy": self.energy,
            "tags": list(self.tags),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MidiSelection:
    """LLM-selected MIDI file with the image context used."""

    path: Path
    filename: str
    reason: str
    image_description: str
    mood: str
    features: dict[str, float | int]


def list_midi_files(
    midi_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> list[MidiCandidate]:
    """Return local MIDI files sorted by filename, enriched by a manifest."""

    root = Path(midi_dir).expanduser()
    if not root.exists():
        return []

    manifest = load_midi_manifest(root, manifest_path=manifest_path)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS
    )
    candidates = []
    for index, path in enumerate(files, start=1):
        metadata = _metadata_for_path(path, root, manifest)
        candidates.append(
            MidiCandidate(
                index=index,
                path=path,
                title=str(metadata.get("title") or ""),
                composer=str(metadata.get("composer") or ""),
                style=str(metadata.get("style") or ""),
                mood=str(metadata.get("mood") or ""),
                energy=str(metadata.get("energy") or ""),
                tags=tuple(str(tag) for tag in metadata.get("tags", []) if str(tag).strip()),
                notes=str(metadata.get("notes") or ""),
            )
        )
    return candidates


def load_midi_manifest(
    midi_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load ``midi_manifest.json`` from the MIDI folder when present."""

    root = Path(midi_dir).expanduser()
    path = Path(manifest_path).expanduser() if manifest_path else root / DEFAULT_MANIFEST_NAME
    if not path.exists():
        return {"tracks": []}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return {"tracks": data}
    if not isinstance(data, dict):
        raise ValueError(f"MIDI manifest must be a JSON object or array: {path}")
    tracks = data.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError(f"MIDI manifest 'tracks' must be a list: {path}")
    return data


def choose_midi_for_image(
    image: np.ndarray,
    midi_dir: str | Path,
    *,
    features: dict[str, float | int] | None = None,
    model: str | None = None,
    max_candidates: int = 80,
    manifest_path: str | Path | None = None,
) -> MidiSelection:
    """Describe ``image`` and choose the best matching local MIDI file."""

    candidates = list_midi_files(midi_dir, manifest_path=manifest_path)
    if not candidates:
        raise FileNotFoundError(f"No MIDI files found in {Path(midi_dir).expanduser()}")

    description = describe_image(image, include_parts=True, max_parts=4)
    return choose_midi_from_description(
        description.description,
        candidates,
        features=features or description.features,
        model=model,
        max_candidates=max_candidates,
    )


def choose_midi_from_description(
    image_description: str,
    candidates: list[MidiCandidate],
    *,
    features: dict[str, float | int] | None = None,
    model: str | None = None,
    max_candidates: int = 80,
) -> MidiSelection:
    """Ask OpenAI to choose one MIDI candidate from a local library."""

    if not candidates:
        raise ValueError("No MIDI candidates were provided")

    limited = _candidate_subset(candidates, max_candidates)
    print(
        f"[midi-selector] candidates total={len(candidates)} sent_to_openai={len(limited)}",
        flush=True,
    )
    print(
        "[midi-selector] sample="
        + ", ".join(f"{candidate.index}:{candidate.filename}" for candidate in limited[:8]),
        flush=True,
    )
    feature_summary = _round_features(features or {})
    prompt = _build_selection_prompt(image_description, limited, feature_summary)
    response = _call_openai_text(prompt, model=model or DEFAULT_OPENAI_TEXT_MODEL)
    choice = _parse_choice(response)

    selected = _candidate_by_filename(limited, choice.get("filename", ""))
    if selected is None:
        selected = _candidate_by_index(limited, choice.get("index"))
    if selected is None:
        selected = _heuristic_choice(limited, feature_summary)

    return MidiSelection(
        path=selected.path,
        filename=selected.filename,
        reason=str(choice.get("reason") or "Selected from the available MIDI library."),
        image_description=image_description,
        mood=str(choice.get("mood") or "image matched"),
        features=feature_summary,
    )


def _candidate_subset(candidates: list[MidiCandidate], max_candidates: int) -> list[MidiCandidate]:
    """Build a mixed candidate set instead of taking the first alphabetical files."""

    if len(candidates) <= max_candidates:
        return candidates

    groups: dict[str, list[MidiCandidate]] = {}
    for candidate in candidates:
        key = candidate.style or candidate.mood or "unknown"
        groups.setdefault(key, []).append(candidate)

    rng = random.SystemRandom()
    for group in groups.values():
        rng.shuffle(group)

    subset: list[MidiCandidate] = []
    keys = list(groups)
    rng.shuffle(keys)
    while len(subset) < max_candidates and keys:
        next_keys = []
        for key in keys:
            group = groups[key]
            if group and len(subset) < max_candidates:
                subset.append(group.pop())
            if group:
                next_keys.append(key)
        keys = next_keys

    return sorted(subset, key=lambda candidate: candidate.index)


def _build_selection_prompt(
    image_description: str,
    candidates: list[MidiCandidate],
    features: dict[str, float | int],
) -> str:
    candidate_text = json.dumps([candidate.summary() for candidate in candidates], indent=2)
    feature_text = json.dumps(features, indent=2)
    return (
        "You are choosing one existing MIDI file to play through FluidSynth. "
        "Choose from the candidate list only. Match the image's visual mood, energy, "
        "brightness, color, subject, and complexity to the filename/label. "
        "Return only JSON with keys: index, filename, mood, reason.\n\n"
        f"Image description:\n{image_description}\n\n"
        f"OpenCV features:\n{feature_text}\n\n"
        f"Available MIDI files:\n{candidate_text}"
    )


def _call_openai_text(prompt: str, *, model: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for MIDI selection")

    payload = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": 500,
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI MIDI selection failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI MIDI selection failed: {exc.reason}") from exc

    text = _extract_response_text(response_data)
    if not text:
        raise RuntimeError(f"OpenAI MIDI selection returned no text: {response_data}")
    return text


def _extract_response_text(response_data: dict[str, Any]) -> str:
    output_text = response_data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for output in response_data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _parse_choice(response_text: str) -> dict[str, Any]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise RuntimeError(f"OpenAI MIDI selection did not return JSON: {response_text}")


def _candidate_by_filename(candidates: list[MidiCandidate], filename: str) -> MidiCandidate | None:
    if not filename:
        return None
    requested = Path(filename).name.lower()
    for candidate in candidates:
        if candidate.filename.lower() == requested:
            return candidate
    return None


def _candidate_by_index(candidates: list[MidiCandidate], index: Any) -> MidiCandidate | None:
    try:
        selected_index = int(index)
    except (TypeError, ValueError):
        return None
    for candidate in candidates:
        if candidate.index == selected_index:
            return candidate
    return None


def _heuristic_choice(candidates: list[MidiCandidate], features: dict[str, float | int]) -> MidiCandidate:
    brightness = float(features.get("brightness", 128))
    saturation = float(features.get("saturation", 80))
    motion = float(features.get("motion", 0))
    keywords = ["minor", "nocturne", "moon", "adagio"]
    if brightness > 170 or saturation > 140 or motion > 8:
        keywords = ["dance", "allegro", "march", "jig", "rag", "bright"]
    elif brightness > 105:
        keywords = ["sonata", "prelude", "suite", "waltz"]

    for keyword in keywords:
        for candidate in candidates:
            searchable = " ".join(
                [
                    candidate.label,
                    candidate.style,
                    candidate.mood,
                    candidate.energy,
                    " ".join(candidate.tags),
                    candidate.notes,
                ]
            ).lower()
            if keyword in searchable:
                return candidate
    return candidates[0]


def _metadata_for_path(path: Path, root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    tracks = manifest.get("tracks", [])
    if not isinstance(tracks, list):
        return {}

    relative = path.relative_to(root).as_posix()
    filename = path.name
    for item in tracks:
        if not isinstance(item, dict):
            continue
        item_file = str(item.get("file") or item.get("filename") or "").strip()
        if item_file in {relative, filename, path.as_posix()}:
            return item
    return {}


def _round_features(features: dict[str, float | int]) -> dict[str, float | int]:
    return {
        key: round(value, 3) if isinstance(value, float) else value
        for key, value in features.items()
    }


def _filename_to_label(stem: str) -> str:
    return re.sub(r"[_-]+", " ", stem).strip()
