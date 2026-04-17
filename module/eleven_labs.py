"""ElevenLabs Music generation helpers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ELEVENLABS_MUSIC_MODEL = os.getenv("ELEVENLABS_MUSIC_MODEL", "music_v1")
DEFAULT_OUTPUT_FORMAT = os.getenv("ELEVENLABS_MUSIC_OUTPUT_FORMAT", "mp3_44100_128")
ELEVENLABS_MUSIC_URL = "https://api.elevenlabs.io/v1/music"


@dataclass(frozen=True)
class ElevenLabsMusicResult:
    """Generated music and response metadata from ElevenLabs."""

    prompt: str
    model_id: str
    output_format: str
    mime_type: str
    audio: bytes
    song_id: str | None
    metadata: dict[str, Any]

    @property
    def extension(self) -> str:
        codec = self.output_format.split("_", 1)[0]
        if codec == "mp3":
            return ".mp3"
        if codec in {"pcm", "wav"}:
            return ".wav"
        if codec == "ulaw":
            return ".ulaw"
        return ".bin"


def build_elevenlabs_prompt(
    image_description: str,
    *,
    features: dict[str, float | int] | None = None,
    duration: str = "30 seconds",
    instrumental_only: bool = False,
) -> str:
    """Turn an image description and OpenCV features into an ElevenLabs prompt."""

    mood_terms = _feature_mood_terms(features or {})
    vocal_instruction = (
        "Instrumental only, no vocals."
        if instrumental_only
        else "Vocals are optional. If vocals are used, write short original lyrics inspired by the image."
    )
    return (
        f"Create a {duration} song inspired by this camera image. "
        f"Visual description: {image_description.strip()} "
        f"Musical direction: {mood_terms}. "
        "Avoid naming specific copyrighted artists, bands, or songs. "
        "Use a clear structure with intro, main section, and ending. "
        f"{vocal_instruction}"
    )


def generate_elevenlabs_music(
    prompt: str,
    *,
    api_key: str | None = None,
    model_id: str | None = None,
    output_format: str | None = None,
    music_length_ms: int | None = 30000,
    force_instrumental: bool = False,
    seed: int | None = None,
    sign_with_c2pa: bool = False,
    timeout: float = 180.0,
) -> ElevenLabsMusicResult:
    """Generate music with ElevenLabs' ``/v1/music`` endpoint."""

    key = api_key or os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs Music")

    selected_model = model_id or DEFAULT_ELEVENLABS_MUSIC_MODEL
    selected_format = output_format or DEFAULT_OUTPUT_FORMAT

    payload: dict[str, Any] = {
        "prompt": prompt,
        "model_id": selected_model,
        "force_instrumental": force_instrumental,
        "sign_with_c2pa": sign_with_c2pa,
    }
    if music_length_ms is not None:
        payload["music_length_ms"] = music_length_ms
    if seed is not None:
        payload["seed"] = seed

    query = urllib.parse.urlencode({"output_format": selected_format})
    request = urllib.request.Request(
        f"{ELEVENLABS_MUSIC_URL}?{query}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "xi-api-key": key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            audio = response.read()
            headers = response.headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs Music request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ElevenLabs Music request failed: {exc.reason}") from exc

    if not audio:
        raise RuntimeError("ElevenLabs Music response did not include audio data")

    return ElevenLabsMusicResult(
        prompt=prompt,
        model_id=selected_model,
        output_format=selected_format,
        mime_type=_mime_type_for_output_format(selected_format, headers.get("Content-Type")),
        audio=audio,
        song_id=headers.get("song-id"),
        metadata={
            "request_id": headers.get("request-id"),
            "character_count": headers.get("x-character-count"),
            "content_type": headers.get("Content-Type"),
        },
    )


def save_elevenlabs_music_result(
    result: ElevenLabsMusicResult,
    output_dir: str | Path,
    *,
    stem: str = "elevenlabs_music",
) -> dict[str, Path]:
    """Save generated audio, prompt, and metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_path = output_path / f"{stem}{result.extension}"
    audio_path.write_bytes(result.audio)

    prompt_path = output_path / f"{stem}_prompt.txt"
    prompt_path.write_text(result.prompt, encoding="utf-8")

    metadata = {
        "model_id": result.model_id,
        "output_format": result.output_format,
        "mime_type": result.mime_type,
        "song_id": result.song_id,
        "metadata": result.metadata,
    }
    metadata_path = output_path / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "audio": audio_path,
        "prompt": prompt_path,
        "metadata": metadata_path,
    }


def _feature_mood_terms(features: dict[str, float | int]) -> str:
    brightness = float(features.get("brightness", 128))
    contrast = float(features.get("contrast", 40))
    saturation = float(features.get("saturation", 80))
    edge_density = float(features.get("edge_density", 0.08))
    contour_count = int(features.get("contour_count", 8))
    motion = float(features.get("motion", 0))

    energy = "gentle and spacious"
    if brightness > 175 or motion > 10 or edge_density > 0.12:
        energy = "bright, energetic, and rhythmic"
    elif brightness < 75:
        energy = "dark, intimate, and minimal"

    color = "warm organic instrumentation"
    if saturation > 150:
        color = "vivid synth textures and bold melodic colors"
    elif saturation < 45:
        color = "soft piano, muted strings, and restrained percussion"

    arrangement = "simple arrangement"
    if contrast > 60 or contour_count > 20:
        arrangement = "layered arrangement with dynamic changes"

    return f"{energy}; {color}; {arrangement}"


def _mime_type_for_output_format(output_format: str, content_type: str | None) -> str:
    if content_type:
        return content_type.split(";", 1)[0]
    codec = output_format.split("_", 1)[0]
    if codec == "mp3":
        return "audio/mpeg"
    if codec in {"pcm", "wav"}:
        return "audio/wav"
    if codec == "ulaw":
        return "audio/basic"
    return "application/octet-stream"
