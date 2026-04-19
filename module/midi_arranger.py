"""Build simple multi-layer MIDI arrangements from chord and drum MIDI files."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import music21 as _M21
except ModuleNotFoundError:
    _M21 = None


DEFAULT_TEMPO_BPM = 110
DEFAULT_TARGET_SECONDS = 60
DEFAULT_CHORD_PROGRAM = 0

CHORD_CHANNEL = 0
BASS_CHANNEL = 1
PAD_CHANNEL = 2
LEAD_CHANNEL = 3
DRUM_CHANNEL = 9

FULL_TICKS_PER_BEAT = 480
DEFAULT_BEATS_PER_BAR = 4


@dataclass(frozen=True)
class MidiArrangementResult:
    output_path: Path
    chord_path: Path
    drum_path: Path
    tempo_bpm: int
    target_seconds: int
    chord_repeats: int
    drum_repeats: int


def create_chord_drum_arrangement(
    chord_midi: str | Path,
    drum_midi: str | Path,
    output_path: str | Path,
    *,
    tempo_bpm: int = DEFAULT_TEMPO_BPM,
    target_seconds: int = DEFAULT_TARGET_SECONDS,
    chord_program: int | None = DEFAULT_CHORD_PROGRAM,
) -> MidiArrangementResult:
    """Merge a chord MIDI and a drum MIDI into a repeated arrangement."""

    try:
        import mido
    except ModuleNotFoundError as exc:
        raise RuntimeError("mido is required. Run: pip install -r requirements.txt") from exc

    chord_path = Path(chord_midi).expanduser()
    drum_path = Path(drum_midi).expanduser()
    out_path = Path(output_path).expanduser()
    if not chord_path.exists():
        raise FileNotFoundError(f"Chord MIDI does not exist: {chord_path}")
    if not drum_path.exists():
        raise FileNotFoundError(f"Drum MIDI does not exist: {drum_path}")

    chord_mid = mido.MidiFile(chord_path)
    drum_mid = mido.MidiFile(drum_path)
    ticks_per_beat = max(chord_mid.ticks_per_beat, drum_mid.ticks_per_beat)

    chord_events, chord_length = _extract_events(chord_mid, ticks_per_beat, target_channel=CHORD_CHANNEL)
    drum_events, drum_length = _extract_events(drum_mid, ticks_per_beat, target_channel=DRUM_CHANNEL)

    target_ticks = _seconds_to_ticks(target_seconds, tempo_bpm, ticks_per_beat)
    chord_repeats = _repeat_count(chord_length, target_ticks)
    drum_repeats = _repeat_count(drum_length, target_ticks)

    events = []
    if chord_program is not None:
        events.append((0, mido.Message("program_change", channel=CHORD_CHANNEL, program=chord_program)))

    events.extend(_repeat_events(chord_events, chord_length, chord_repeats, target_ticks))
    events.extend(_repeat_events(drum_events, drum_length, drum_repeats, target_ticks))
    events.sort(key=lambda item: _event_sort_key(item[0], item[1]))

    output = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    output.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="camera_arrangement", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    current_tick = 0
    for absolute_tick, message in events:
        if absolute_tick > target_ticks:
            continue
        message = message.copy(time=max(0, absolute_tick - current_tick))
        current_tick = absolute_tick
        track.append(message)

    track.append(mido.MetaMessage("end_of_track", time=max(0, target_ticks - current_tick)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)

    return MidiArrangementResult(
        output_path=out_path,
        chord_path=chord_path,
        drum_path=drum_path,
        tempo_bpm=tempo_bpm,
        target_seconds=target_seconds,
        chord_repeats=chord_repeats,
        drum_repeats=drum_repeats,
    )


def default_arrangement_path(output_dir: str | Path = "generated_arrangements") -> Path:
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path / f"arrangement_{time.strftime('%Y%m%d-%H%M%S')}.mid"


def _extract_events(midi, target_ticks_per_beat: int, *, target_channel: int) -> tuple[list[tuple[int, object]], int]:
    events = []
    max_tick = 0
    scale = target_ticks_per_beat / midi.ticks_per_beat
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += int(round(message.time * scale))
            max_tick = max(max_tick, absolute)
            if message.is_meta:
                continue
            if message.type in {"note_on", "note_off", "control_change", "program_change", "pitchwheel"}:
                copied = message.copy(time=0)
                if hasattr(copied, "channel"):
                    copied.channel = target_channel
                events.append((absolute, copied))

    return events, max(1, max_tick)


def _repeat_events(
    events: list[tuple[int, object]],
    length_ticks: int,
    repeats: int,
    target_ticks: int,
) -> list[tuple[int, object]]:
    repeated = []
    for repeat in range(repeats):
        offset = repeat * length_ticks
        for tick, message in events:
            absolute = offset + tick
            if absolute <= target_ticks:
                repeated.append((absolute, message))
    return repeated


def _seconds_to_ticks(seconds: int, tempo_bpm: int, ticks_per_beat: int) -> int:
    beats = seconds * tempo_bpm / 60
    return int(beats * ticks_per_beat)


def _repeat_count(length_ticks: int, target_ticks: int) -> int:
    return max(1, int(math.ceil(target_ticks / max(1, length_ticks))))


def _event_sort_key(tick: int, message) -> tuple[int, int]:
    order = 1
    if message.type == "program_change":
        order = 0
    elif message.type == "note_off" or (message.type == "note_on" and getattr(message, "velocity", 1) == 0):
        order = 2
    return tick, order


# ---------------------------------------------------------------------------
# Vibe-driven full arranger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VibeProfile:
    """Vibe-derived knobs from OpenCV features (each in 0..1)."""

    energy: float
    brightness: float
    color: float
    complexity: float


@dataclass(frozen=True)
class LayerPalette:
    """General MIDI program numbers per layer."""

    chord_program: int
    bass_program: int
    pad_program: int
    lead_program: int


@dataclass(frozen=True)
class Harmony:
    """One harmonic frame produced by music21 chordify."""

    offset_ql: float
    duration_ql: float
    root_midi: int
    bass_midi: int
    pitches: tuple[int, ...]
    quality: str
    roman: str


@dataclass(frozen=True)
class HarmonicAnalysis:
    """Key + scale + chord progression extracted from a MIDI file."""

    key_tonic_midi: int
    key_mode: str
    scale_pitches: tuple[int, ...]
    progression: tuple[Harmony, ...]
    beats_per_bar: int


@dataclass(frozen=True)
class ChordEvent:
    """A Harmony rendered into absolute-tick space for a section."""

    start_tick: int
    duration_tick: int
    pitches: tuple[int, ...]
    root_midi: int
    bass_midi: int


@dataclass(frozen=True)
class Section:
    kind: str
    start_bar: int
    length_bars: int
    layers: frozenset[str]
    density: float
    velocity_scale: float


@dataclass(frozen=True)
class FullArrangementResult:
    output_path: Path
    chord_path: Path
    drum_path: Path
    tempo_bpm: int
    target_seconds: int
    sections: tuple[Section, ...]
    palette: LayerPalette
    vibe: VibeProfile
    key_root_midi: int
    key_mode: str
    progression_size: int

    def summary(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "chord_path": str(self.chord_path),
            "drum_path": str(self.drum_path),
            "tempo_bpm": self.tempo_bpm,
            "target_seconds": self.target_seconds,
            "key": f"{self.key_root_midi % 12} {self.key_mode}",
            "vibe": {
                "energy": round(self.vibe.energy, 3),
                "brightness": round(self.vibe.brightness, 3),
                "color": round(self.vibe.color, 3),
                "complexity": round(self.vibe.complexity, 3),
            },
            "palette": {
                "chord": self.palette.chord_program,
                "bass": self.palette.bass_program,
                "pad": self.palette.pad_program,
                "lead": self.palette.lead_program,
            },
            "sections": [
                {
                    "kind": s.kind,
                    "start_bar": s.start_bar,
                    "length_bars": s.length_bars,
                    "layers": sorted(s.layers),
                    "density": round(s.density, 2),
                    "velocity_scale": round(s.velocity_scale, 2),
                }
                for s in self.sections
            ],
            "progression_size": self.progression_size,
        }


# ---------------------------------------------------------------------------
# Vibe + palette
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def vibe_from_features(features: dict[str, float | int] | None) -> VibeProfile:
    f = features or {}
    motion = float(f.get("motion", 0.0))
    edge_density = float(f.get("edge_density", 0.05))
    brightness = float(f.get("brightness", 128.0))
    saturation = float(f.get("saturation", 80.0))
    contour_count = float(f.get("contour_count", 8.0))
    contrast = float(f.get("contrast", 40.0))

    energy = _clamp(motion / 30.0 + edge_density * 4.0)
    brightness_n = _clamp(brightness / 255.0)
    color_n = _clamp(saturation / 255.0)
    complexity = _clamp(contour_count / 40.0 + contrast / 200.0)
    return VibeProfile(
        energy=energy,
        brightness=brightness_n,
        color=color_n,
        complexity=complexity,
    )


def palette_for(vibe: VibeProfile) -> LayerPalette:
    if vibe.brightness > 0.6 and vibe.color > 0.5:
        # bright, saturated -> synth pop palette
        return LayerPalette(chord_program=4, bass_program=38, pad_program=89, lead_program=80)
    if vibe.brightness < 0.3:
        # dark -> piano + warm strings
        return LayerPalette(chord_program=0, bass_program=33, pad_program=49, lead_program=73)
    if vibe.color < 0.25:
        # muted -> Rhodes + acoustic bass + soft pad + flute
        return LayerPalette(chord_program=4, bass_program=32, pad_program=48, lead_program=73)
    # warm neutral default
    return LayerPalette(chord_program=0, bass_program=32, pad_program=48, lead_program=73)


# ---------------------------------------------------------------------------
# Music21 analysis
# ---------------------------------------------------------------------------


def analyse_chord_midi(
    chord_path: str | Path,
    *,
    melody_low: int = 60,
    melody_high: int = 84,
) -> HarmonicAnalysis:
    """Analyse the chord MIDI: detect key, build scale, and chordify the progression."""

    if _M21 is None:
        raise RuntimeError("music21 is required. Run: pip install -r requirements.txt")
    m21 = _M21

    path = Path(chord_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Chord MIDI does not exist: {path}")

    score = m21.converter.parse(str(path))

    try:
        key = score.analyze("key")
    except Exception:
        key = m21.key.Key("C", "major")

    scale = key.getScale()
    try:
        scale_pitches = tuple(
            int(p.midi)
            for p in scale.getPitches(
                m21.pitch.Pitch(midi=melody_low),
                m21.pitch.Pitch(midi=melody_high),
            )
        )
    except Exception:
        scale_pitches = tuple(p + 60 for p in (0, 2, 4, 5, 7, 9, 11, 12))

    time_signatures = list(score.recurse().getElementsByClass(m21.meter.TimeSignature))
    beats_per_bar = (
        int(time_signatures[0].numerator) if time_signatures else DEFAULT_BEATS_PER_BAR
    )

    chordified = score.chordify()
    progression: list[Harmony] = []
    for c in chordified.recurse().getElementsByClass(m21.chord.Chord):
        if not c.pitches or c.quarterLength <= 0:
            continue
        try:
            roman_figure = m21.roman.romanNumeralFromChord(c, key).figure
        except Exception:
            roman_figure = ""
        progression.append(
            Harmony(
                offset_ql=float(c.offset),
                duration_ql=float(c.quarterLength),
                root_midi=int(c.root().midi),
                bass_midi=int(c.bass().midi),
                pitches=tuple(int(p.midi) for p in c.pitches),
                quality=str(c.quality or ""),
                roman=str(roman_figure or ""),
            )
        )

    return HarmonicAnalysis(
        key_tonic_midi=int(key.tonic.midi),
        key_mode=str(key.mode),
        scale_pitches=scale_pitches or tuple(p + 60 for p in (0, 2, 4, 5, 7, 9, 11, 12)),
        progression=tuple(progression),
        beats_per_bar=beats_per_bar,
    )


# ---------------------------------------------------------------------------
# Section planning
# ---------------------------------------------------------------------------


LAYER_PRESETS: dict[str, frozenset[str]] = {
    "intro": frozenset({"pad"}),
    "verse": frozenset({"pad", "chord", "bass", "drums"}),
    "chorus": frozenset({"pad", "chord", "bass", "drums", "lead"}),
    "bridge": frozenset({"pad", "bass", "lead"}),
    "outro": frozenset({"pad", "chord"}),
}
DENSITY_BY_KIND = {"intro": 0.4, "verse": 0.7, "chorus": 1.0, "bridge": 0.6, "outro": 0.3}
VELOCITY_BY_KIND = {"intro": 0.7, "verse": 0.9, "chorus": 1.05, "bridge": 0.85, "outro": 0.65}

FORM_TEMPLATES: list[tuple[int, list[tuple[str, int]]]] = [
    (16, [("intro", 2), ("verse", 6), ("chorus", 6), ("outro", 2)]),
    (32, [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 4), ("chorus", 6), ("outro", 2)]),
    (64, [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8),
          ("bridge", 4), ("chorus", 8), ("chorus", 8), ("outro", 4)]),
]


def _select_template(total_bars: int) -> list[tuple[str, int]]:
    for limit, template in FORM_TEMPLATES:
        if total_bars <= limit:
            return template
    return FORM_TEMPLATES[-1][1]


def _scale_template_to_bars(
    template: list[tuple[str, int]],
    total_bars: int,
) -> list[tuple[str, int]]:
    raw_total = sum(length for _, length in template)
    if raw_total <= 0 or total_bars <= 0:
        return [("verse", max(1, total_bars))]

    scale = total_bars / raw_total
    scaled = [(kind, max(1, int(round(length * scale)))) for kind, length in template]
    diff = total_bars - sum(length for _, length in scaled)
    if diff != 0:
        # Adjust the longest section so the bars sum exactly.
        index = max(range(len(scaled)), key=lambda i: scaled[i][1])
        kind, length = scaled[index]
        scaled[index] = (kind, max(1, length + diff))
    return scaled


def plan_sections(total_bars: int, vibe: VibeProfile) -> list[Section]:
    template = _select_template(total_bars)
    scaled = _scale_template_to_bars(template, total_bars)

    sections: list[Section] = []
    cursor = 0
    for kind, length in scaled:
        layers = set(LAYER_PRESETS[kind])
        if vibe.energy > 0.6 and kind == "verse":
            layers.add("lead")
        if vibe.energy < 0.3 and kind == "chorus":
            layers.discard("lead")
        if vibe.complexity < 0.2:
            layers.discard("pad")
        density = _clamp(DENSITY_BY_KIND[kind] * (0.7 + 0.5 * vibe.energy), 0.15, 1.0)
        sections.append(
            Section(
                kind=kind,
                start_bar=cursor,
                length_bars=length,
                layers=frozenset(layers),
                density=density,
                velocity_scale=VELOCITY_BY_KIND[kind],
            )
        )
        cursor += length
    return sections


# ---------------------------------------------------------------------------
# Render the chord progression into a section (absolute ticks)
# ---------------------------------------------------------------------------


def render_section(
    section: Section,
    analysis: HarmonicAnalysis,
    *,
    ticks_per_beat: int = FULL_TICKS_PER_BEAT,
) -> list[ChordEvent]:
    """Loop the source progression to fill the section in absolute-tick space."""

    if not analysis.progression:
        return []

    beats_per_bar = analysis.beats_per_bar or DEFAULT_BEATS_PER_BAR
    section_start_tick = section.start_bar * beats_per_bar * ticks_per_beat
    section_end_tick = (section.start_bar + section.length_bars) * beats_per_bar * ticks_per_beat

    events: list[ChordEvent] = []
    cursor_tick = section_start_tick
    progression = list(analysis.progression)
    index = 0
    while cursor_tick < section_end_tick:
        harmony = progression[index % len(progression)]
        duration_tick = max(1, int(round(harmony.duration_ql * ticks_per_beat)))
        end_tick = min(section_end_tick, cursor_tick + duration_tick)
        events.append(
            ChordEvent(
                start_tick=cursor_tick,
                duration_tick=end_tick - cursor_tick,
                pitches=harmony.pitches,
                root_midi=harmony.root_midi,
                bass_midi=harmony.bass_midi,
            )
        )
        cursor_tick = end_tick
        index += 1
    return events


# ---------------------------------------------------------------------------
# Layer builders (return lists of (absolute_tick, mido.Message))
# ---------------------------------------------------------------------------


def _bass_register(pitch: int, low: int = 36, high: int = 55) -> int:
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch


def _pad_register(pitch: int, low: int = 55, high: int = 76) -> int:
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch


def build_chord_layer(
    chords: list[ChordEvent],
    palette: LayerPalette,
    vibe: VibeProfile,
):
    import mido

    if not chords:
        return []
    events = [(0, mido.Message("program_change", channel=CHORD_CHANNEL, program=palette.chord_program))]
    velocity = int(60 + vibe.energy * 35)
    for chord in chords:
        for pitch in chord.pitches:
            events.append(
                (chord.start_tick,
                 mido.Message("note_on", channel=CHORD_CHANNEL, note=pitch, velocity=velocity))
            )
            events.append(
                (chord.start_tick + max(1, chord.duration_tick - 4),
                 mido.Message("note_off", channel=CHORD_CHANNEL, note=pitch, velocity=0))
            )
    return events


def build_bass_layer(
    chords: list[ChordEvent],
    palette: LayerPalette,
    vibe: VibeProfile,
    *,
    ticks_per_beat: int = FULL_TICKS_PER_BEAT,
):
    import mido

    if not chords:
        return []
    events = [(0, mido.Message("program_change", channel=BASS_CHANNEL, program=palette.bass_program))]
    velocity = int(80 + vibe.energy * 30)

    walk = vibe.energy > 0.5
    for chord in chords:
        root = _bass_register(chord.bass_midi or chord.root_midi)
        if not walk:
            events.append(
                (chord.start_tick,
                 mido.Message("note_on", channel=BASS_CHANNEL, note=root, velocity=velocity))
            )
            events.append(
                (chord.start_tick + max(1, chord.duration_tick - 4),
                 mido.Message("note_off", channel=BASS_CHANNEL, note=root, velocity=0))
            )
            continue

        beat_tick = max(1, ticks_per_beat // 2)
        cursor = chord.start_tick
        toggle = 0
        while cursor < chord.start_tick + chord.duration_tick:
            pitch = root if toggle % 2 == 0 else _bass_register(root + 7)
            end = min(chord.start_tick + chord.duration_tick, cursor + beat_tick)
            events.append(
                (cursor,
                 mido.Message("note_on", channel=BASS_CHANNEL, note=pitch, velocity=velocity))
            )
            events.append(
                (max(cursor + 1, end - 4),
                 mido.Message("note_off", channel=BASS_CHANNEL, note=pitch, velocity=0))
            )
            cursor = end
            toggle += 1
    return events


def build_pad_layer(
    chords: list[ChordEvent],
    palette: LayerPalette,
    vibe: VibeProfile,
):
    import mido

    if not chords:
        return []
    events = [(0, mido.Message("program_change", channel=PAD_CHANNEL, program=palette.pad_program))]
    velocity = int(40 + vibe.color * 30)
    for chord in chords:
        voiced = sorted({_pad_register(p) for p in chord.pitches})
        for pitch in voiced:
            events.append(
                (chord.start_tick,
                 mido.Message("note_on", channel=PAD_CHANNEL, note=pitch, velocity=velocity))
            )
            events.append(
                (chord.start_tick + max(1, chord.duration_tick - 2),
                 mido.Message("note_off", channel=PAD_CHANNEL, note=pitch, velocity=0))
            )
    return events


# Rhythmic motifs in beats: list of (start_beat, duration_beat, role)
# role in {"chord_tone", "passing", "ornament", "rest"}
MELODY_MOTIFS: dict[str, list[list[tuple[float, float, str]]]] = {
    "low": [
        [(0.0, 4.0, "chord_tone")],
        [(0.0, 2.0, "chord_tone"), (2.0, 2.0, "passing")],
    ],
    "mid": [
        [(0.0, 1.0, "chord_tone"), (1.0, 1.0, "passing"),
         (2.0, 1.0, "chord_tone"), (3.0, 1.0, "rest")],
        [(0.0, 1.5, "chord_tone"), (1.5, 0.5, "ornament"),
         (2.0, 1.0, "chord_tone"), (3.0, 1.0, "passing")],
    ],
    "high": [
        [(b * 0.5, 0.5, "chord_tone" if b % 2 == 0 else "passing") for b in range(8)],
        [(0.0, 0.5, "chord_tone"), (0.5, 0.5, "passing"),
         (1.0, 0.5, "chord_tone"), (1.5, 0.5, "ornament"),
         (2.0, 0.5, "chord_tone"), (2.5, 0.5, "passing"),
         (3.0, 1.0, "chord_tone")],
    ],
}


def _energy_bucket(vibe: VibeProfile) -> str:
    if vibe.energy > 0.6:
        return "high"
    if vibe.energy > 0.3:
        return "mid"
    return "low"


def _closest(options: list[int], target: int | None) -> int:
    if target is None:
        return options[len(options) // 2]
    return min(options, key=lambda p: abs(p - target))


def build_melody_layer(
    chords: list[ChordEvent],
    analysis: HarmonicAnalysis,
    palette: LayerPalette,
    vibe: VibeProfile,
    *,
    rng: random.Random,
    ticks_per_beat: int = FULL_TICKS_PER_BEAT,
):
    import mido

    if not chords or not analysis.scale_pitches:
        return []

    events = [(0, mido.Message("program_change", channel=LEAD_CHANNEL, program=palette.lead_program))]
    bucket = _energy_bucket(vibe)
    motifs = MELODY_MOTIFS[bucket]
    scale = list(analysis.scale_pitches)
    last_pitch: int | None = None

    for chord in chords:
        chord_pcs = {p % 12 for p in chord.pitches}
        chord_tone_options = [p for p in scale if p % 12 in chord_pcs]
        if not chord_tone_options:
            chord_tone_options = scale

        chord_beats = chord.duration_tick / ticks_per_beat
        motif = rng.choice(motifs)
        motif_beats = sum(d for _, d, role in motif if role != "rest") + sum(
            d for _, d, role in motif if role == "rest"
        )
        scale_factor = chord_beats / motif_beats if motif_beats > 0 else 1.0

        for beat_offset, beat_dur, role in motif:
            if role == "rest":
                continue
            if role == "chord_tone":
                options = chord_tone_options
            else:
                options = scale

            pitch = _closest(options, last_pitch) if last_pitch else rng.choice(options)
            last_pitch = pitch

            start_tick = chord.start_tick + int(round(beat_offset * scale_factor * ticks_per_beat))
            dur_tick = max(2, int(round(beat_dur * scale_factor * ticks_per_beat)) - 2)
            end_tick = min(chord.start_tick + chord.duration_tick - 1, start_tick + dur_tick)
            if end_tick <= start_tick:
                continue
            velocity = int(70 + vibe.energy * 30 + (10 if role == "chord_tone" else 0))
            events.append(
                (start_tick,
                 mido.Message("note_on", channel=LEAD_CHANNEL, note=pitch, velocity=velocity))
            )
            events.append(
                (end_tick,
                 mido.Message("note_off", channel=LEAD_CHANNEL, note=pitch, velocity=0))
            )
    return events


def build_drum_section(
    drum_events: list[tuple[int, Any]],
    drum_length: int,
    section: Section,
    *,
    beats_per_bar: int,
    ticks_per_beat: int = FULL_TICKS_PER_BEAT,
):
    import mido

    if not drum_events or drum_length <= 0:
        return []

    section_start_tick = section.start_bar * beats_per_bar * ticks_per_beat
    section_end_tick = (section.start_bar + section.length_bars) * beats_per_bar * ticks_per_beat
    section_length = section_end_tick - section_start_tick
    repeats = max(1, math.ceil(section_length / drum_length))

    events: list[tuple[int, Any]] = []
    for repeat in range(repeats):
        offset = section_start_tick + repeat * drum_length
        for tick, message in drum_events:
            absolute = offset + tick
            if absolute >= section_end_tick:
                break
            new_message = message.copy(time=0)
            if hasattr(new_message, "channel"):
                new_message.channel = DRUM_CHANNEL
            # dmp_midi's generic "Cymbal" was imported as GM 49 (Crash Cymbal 1)
            # but is actually the steady time-keeping cymbal — remap to ride.
            if getattr(new_message, "note", None) == 49:
                new_message.note = 51
            events.append((absolute, new_message))
    return events


# ---------------------------------------------------------------------------
# Density + velocity post-processing
# ---------------------------------------------------------------------------


def _apply_density(
    events: list[tuple[int, Any]],
    density: float,
    rng: random.Random,
) -> list[tuple[int, Any]]:
    if density >= 0.999:
        return events
    keep_notes: dict[tuple[int, int], bool] = {}
    out: list[tuple[int, Any]] = []
    for tick, message in events:
        if message.type not in ("note_on", "note_off"):
            out.append((tick, message))
            continue
        key = (getattr(message, "channel", -1), getattr(message, "note", -1))
        if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
            keep = rng.random() < density
            keep_notes[key] = keep
            if keep:
                out.append((tick, message))
        else:
            if keep_notes.pop(key, True):
                out.append((tick, message))
    return out


def _scale_velocity(events: list[tuple[int, Any]], scale: float) -> list[tuple[int, Any]]:
    if abs(scale - 1.0) < 1e-3:
        return events
    out: list[tuple[int, Any]] = []
    for tick, message in events:
        if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
            new_velocity = max(1, min(127, int(round(message.velocity * scale))))
            out.append((tick, message.copy(velocity=new_velocity)))
        else:
            out.append((tick, message))
    return out


def _seed_from_vibe(vibe: VibeProfile, salt: str = "") -> int:
    base = (
        int(vibe.energy * 1000) * 31
        + int(vibe.brightness * 1000) * 37
        + int(vibe.color * 1000) * 41
        + int(vibe.complexity * 1000) * 43
    )
    for ch in salt:
        base = base * 131 + ord(ch)
    return base & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def create_full_arrangement(
    chord_midi: str | Path,
    drum_midi: str | Path,
    output_path: str | Path,
    *,
    features: dict[str, float | int] | None = None,
    tempo_bpm: int = DEFAULT_TEMPO_BPM,
    target_seconds: int = DEFAULT_TARGET_SECONDS,
    ticks_per_beat: int = FULL_TICKS_PER_BEAT,
) -> FullArrangementResult:
    """Build a vibe-driven, multi-layer arrangement with verse/chorus structure."""

    try:
        import mido
    except ModuleNotFoundError as exc:
        raise RuntimeError("mido is required. Run: pip install -r requirements.txt") from exc

    chord_path = Path(chord_midi).expanduser()
    drum_path = Path(drum_midi).expanduser()
    out_path = Path(output_path).expanduser()
    if not drum_path.exists():
        raise FileNotFoundError(f"Drum MIDI does not exist: {drum_path}")

    analysis = analyse_chord_midi(chord_path)
    vibe = vibe_from_features(features)
    palette = palette_for(vibe)

    beats_per_bar = analysis.beats_per_bar or DEFAULT_BEATS_PER_BAR
    total_beats = max(1, int(round(target_seconds * tempo_bpm / 60)))
    total_bars = max(1, total_beats // beats_per_bar)
    sections = plan_sections(total_bars, vibe)

    drum_mid = mido.MidiFile(drum_path)
    drum_events, drum_length = _extract_events(drum_mid, ticks_per_beat, target_channel=DRUM_CHANNEL)

    all_events: list[tuple[int, Any]] = []

    for section in sections:
        section_chords = render_section(section, analysis, ticks_per_beat=ticks_per_beat)

        builders = {
            "chord": lambda: build_chord_layer(section_chords, palette, vibe),
            "bass":  lambda: build_bass_layer(section_chords, palette, vibe, ticks_per_beat=ticks_per_beat),
            "pad":   lambda: build_pad_layer(section_chords, palette, vibe),
            "lead":  lambda: build_melody_layer(
                section_chords, analysis, palette, vibe,
                rng=random.Random(_seed_from_vibe(vibe, f"{section.kind}-{section.start_bar}-lead")),
                ticks_per_beat=ticks_per_beat,
            ),
            "drums": lambda: build_drum_section(
                drum_events, drum_length, section,
                beats_per_bar=beats_per_bar, ticks_per_beat=ticks_per_beat,
            ),
        }

        for layer_name in section.layers:
            builder = builders.get(layer_name)
            if builder is None:
                continue
            layer_events = builder()
            if not layer_events:
                continue
            density_rng = random.Random(_seed_from_vibe(vibe, f"{section.kind}-{section.start_bar}-{layer_name}"))
            layer_events = _apply_density(layer_events, section.density, density_rng)
            layer_events = _scale_velocity(layer_events, section.velocity_scale)
            all_events.extend(layer_events)

    all_events.sort(key=lambda item: _event_sort_key(item[0], item[1]))

    output = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    output.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="vibe_arrangement", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=beats_per_bar, denominator=4, time=0))

    end_tick = total_bars * beats_per_bar * ticks_per_beat
    current_tick = 0
    for absolute_tick, message in all_events:
        if absolute_tick > end_tick:
            continue
        track.append(message.copy(time=max(0, absolute_tick - current_tick)))
        current_tick = absolute_tick

    track.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - current_tick)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)

    return FullArrangementResult(
        output_path=out_path,
        chord_path=chord_path,
        drum_path=drum_path,
        tempo_bpm=tempo_bpm,
        target_seconds=target_seconds,
        sections=tuple(sections),
        palette=palette,
        vibe=vibe,
        key_root_midi=analysis.key_tonic_midi,
        key_mode=analysis.key_mode,
        progression_size=len(analysis.progression),
    )
