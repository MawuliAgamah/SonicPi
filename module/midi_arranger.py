"""Build simple multi-layer MIDI arrangements from chord and drum MIDI files."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TEMPO_BPM = 110
DEFAULT_TARGET_SECONDS = 60
DEFAULT_CHORD_PROGRAM = 0
CHORD_CHANNEL = 0
DRUM_CHANNEL = 9


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
