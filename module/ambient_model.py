"""Pure modal music helpers for camera-driven ambient synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .midi_arranger import VibeProfile, vibe_from_features


MODE_INTERVALS: dict[str, tuple[int, ...]] = {
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
}

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class ModalKey:
    tonic_midi: int
    mode_name: str

    @property
    def label(self) -> str:
        return f"{midi_note_name(self.tonic_midi)} {self.mode_name.title()}"


@dataclass(frozen=True)
class AmbientChord:
    root_midi: int
    scale_degrees: tuple[int, ...]
    pitches: tuple[int, ...]
    roman: str
    name: str


@dataclass(frozen=True)
class AmbientMusicState:
    vibe: VibeProfile
    key: ModalKey
    chord: AmbientChord
    tempo_bpm: int
    chord_change_seconds: float
    layers: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "vibe": {
                "energy": round(self.vibe.energy, 3),
                "brightness": round(self.vibe.brightness, 3),
                "color": round(self.vibe.color, 3),
                "complexity": round(self.vibe.complexity, 3),
            },
            "key": self.key.label,
            "chord": {
                "name": self.chord.name,
                "roman": self.chord.roman,
                "root": midi_note_name(self.chord.root_midi),
                "pitches": [midi_note_name(pitch) for pitch in self.chord.pitches],
            },
            "tempo_bpm": self.tempo_bpm,
            "chord_change_seconds": round(self.chord_change_seconds, 2),
            "layers": list(self.layers),
        }


def midi_note_name(note: int) -> str:
    octave = note // 12 - 1
    return f"{NOTE_NAMES[note % 12]}{octave}"


def smooth_vibe(
    previous: VibeProfile | None,
    current: VibeProfile,
    *,
    alpha: float,
) -> VibeProfile:
    """Exponential smoothing for vibe values."""

    alpha = _clamp(alpha)
    if previous is None:
        return current

    return VibeProfile(
        energy=_lerp(previous.energy, current.energy, alpha),
        brightness=_lerp(previous.brightness, current.brightness, alpha),
        color=_lerp(previous.color, current.color, alpha),
        complexity=_lerp(previous.complexity, current.complexity, alpha),
    )


def pick_modal_key(vibe: VibeProfile) -> ModalKey:
    """Pick one stable modal key from the starting vibe."""

    if vibe.brightness > 0.64:
        mode = "lydian"
    elif vibe.brightness < 0.30:
        mode = "phrygian"
    elif vibe.color < 0.22:
        mode = "aeolian"
    else:
        mode = "dorian"

    tonics = (38, 41, 43, 45, 50, 53, 55)  # D2, F2, G2, A2, D3, F3, G3
    index = int(_clamp(vibe.color * 0.55 + vibe.complexity * 0.45) * (len(tonics) - 1))
    tonic = tonics[index]
    if vibe.brightness > 0.68:
        tonic += 12
    elif vibe.brightness < 0.24:
        tonic -= 12
    return ModalKey(tonic_midi=max(24, min(72, tonic)), mode_name=mode)


def modal_scale_pitches(
    key: ModalKey,
    *,
    low: int = 36,
    high: int = 96,
) -> list[int]:
    intervals = MODE_INTERVALS.get(key.mode_name, MODE_INTERVALS["dorian"])
    pitches: list[int] = []
    root = key.tonic_midi
    while root > low:
        root -= 12
    while root <= high:
        for interval in intervals:
            pitch = root + interval
            if low <= pitch <= high:
                pitches.append(pitch)
        root += 12
    return sorted(set(pitches))


def generate_chord_progression(
    key: ModalKey,
    vibe: VibeProfile,
    *,
    length: int = 4,
) -> list[AmbientChord]:
    """Generate slow modal chords without dominant cadences."""

    if key.mode_name == "lydian":
        roots = (0, 2, 1, 4)
        romans = ("Imaj9", "iii7", "II", "Vmaj7")
    elif key.mode_name == "phrygian":
        roots = (0, 1, 3, 6)
        romans = ("i", "bIImaj7", "iv7", "bVII")
    elif key.mode_name == "aeolian":
        roots = (0, 5, 3, 6)
        romans = ("i", "VImaj7", "iv7", "bVII")
    else:
        roots = (0, 3, 1, 6)
        romans = ("i9", "IV", "ii7", "bVII")

    chords = [
        _build_chord(key, root_degree=root, roman=roman, complexity=vibe.complexity)
        for root, roman in zip(roots, romans, strict=False)
    ]
    return chords[: max(1, length)]


def chord_for_time(
    key: ModalKey,
    vibe: VibeProfile,
    elapsed_seconds: float,
) -> AmbientChord:
    progression = generate_chord_progression(key, vibe)
    interval = chord_change_seconds(vibe)
    index = int(max(0.0, elapsed_seconds) // interval) % len(progression)
    return progression[index]


def chord_change_seconds(vibe: VibeProfile) -> float:
    """Complex scenes move through harmony sooner, but still slowly."""

    return _lerp(32.0, 12.0, vibe.complexity)


def tempo_from_vibe(vibe: VibeProfile) -> int:
    return int(round(_lerp(60.0, 80.0, vibe.energy * 0.65 + vibe.brightness * 0.35)))


def active_layers(vibe: VibeProfile, *, enable_arpeggio: bool = True, enable_texture: bool = True) -> tuple[str, ...]:
    layers = ["drone", "pad"]
    if enable_arpeggio and vibe.energy > 0.14:
        layers.append("arpeggio")
    if enable_texture and (vibe.color > 0.18 or vibe.energy > 0.28):
        layers.append("texture")
    return tuple(layers)


def ambient_state_for(
    vibe: VibeProfile,
    key: ModalKey,
    *,
    elapsed_seconds: float = 0.0,
    enable_arpeggio: bool = True,
    enable_texture: bool = True,
) -> AmbientMusicState:
    return AmbientMusicState(
        vibe=vibe,
        key=key,
        chord=chord_for_time(key, vibe, elapsed_seconds),
        tempo_bpm=tempo_from_vibe(vibe),
        chord_change_seconds=chord_change_seconds(vibe),
        layers=active_layers(vibe, enable_arpeggio=enable_arpeggio, enable_texture=enable_texture),
    )


def vibe_from_feature_dict(features: dict[str, float | int] | None) -> VibeProfile:
    return vibe_from_features(features)


def _build_chord(
    key: ModalKey,
    *,
    root_degree: int,
    roman: str,
    complexity: float,
) -> AmbientChord:
    intervals = MODE_INTERVALS.get(key.mode_name, MODE_INTERVALS["dorian"])
    degree_offsets = [0, 2, 4]
    if complexity > 0.22:
        degree_offsets.append(6)
    if complexity > 0.55:
        degree_offsets.append(8)

    scale_degrees = tuple(root_degree + offset for offset in degree_offsets)
    pitches = tuple(_degree_to_pitch(key.tonic_midi, intervals, degree) for degree in scale_degrees)
    root = pitches[0]
    name = f"{midi_note_name(root)} {roman}"
    return AmbientChord(
        root_midi=root,
        scale_degrees=scale_degrees,
        pitches=pitches,
        roman=roman,
        name=name,
    )


def _degree_to_pitch(tonic_midi: int, intervals: Iterable[int], degree: int) -> int:
    scale = tuple(intervals)
    octave, index = divmod(degree, len(scale))
    return tonic_midi + octave * 12 + scale[index]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
