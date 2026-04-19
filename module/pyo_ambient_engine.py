"""Continuous pyo synthesis engine for camera-driven ambient sound."""

from __future__ import annotations

import importlib
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from .ambient_model import (
    AmbientMusicState,
    ModalKey,
    ambient_state_for,
    modal_scale_pitches,
    pick_modal_key,
)
from .midi_arranger import VibeProfile


@dataclass(frozen=True)
class PyoEngineStatus:
    running: bool
    key: str
    chord: str
    roman: str
    layers: tuple[str, ...]
    tempo_bpm: int
    message: str


class PyoAmbientEngine:
    """A long-running ambient instrument controlled by smoothed camera vibe."""

    def __init__(
        self,
        *,
        initial_vibe: VibeProfile | None = None,
        key: ModalKey | None = None,
        enable_arpeggio: bool = True,
        enable_texture: bool = True,
        master_gain: float = 0.55,
    ):
        self.vibe = initial_vibe or VibeProfile(
            energy=0.25,
            brightness=0.5,
            color=0.35,
            complexity=0.25,
        )
        self.key = key or pick_modal_key(self.vibe)
        self.enable_arpeggio = enable_arpeggio
        self.enable_texture = enable_texture
        self.master_gain = master_gain

        self._pyo: Any | None = None
        self._server: Any | None = None
        self._pattern: Any | None = None
        self._objects: dict[str, Any] = {}
        self._state = ambient_state_for(
            self.vibe,
            self.key,
            enable_arpeggio=enable_arpeggio,
            enable_texture=enable_texture,
        )
        self._start_time = 0.0
        self._last_motion = 0.0
        self._last_message = "Stopped."
        self._rng = random.Random()
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        with self._lock:
            if self.running:
                return

            self._pyo = _import_pyo()
            self._server = self._pyo.Server().boot()
            self._server.start()
            self._start_time = time.monotonic()
            self._build_graph()
            self._apply_state()
            self._pattern = self._pyo.Pattern(self._play_arpeggio_note, time=1.0).play()
            self._last_message = "Ambient engine running."

    def stop(self) -> None:
        with self._lock:
            if self._pattern is not None:
                try:
                    self._pattern.stop()
                except Exception:
                    pass
                self._pattern = None
            if self._server is not None:
                try:
                    self._server.stop()
                    self._server.shutdown()
                except Exception:
                    pass
                self._server = None
            self._objects.clear()
            self._last_message = "Stopped."

    def set_vibe(self, vibe: VibeProfile, *, motion: float = 0.0) -> None:
        with self._lock:
            self.vibe = vibe
            self._last_motion = max(0.0, float(motion))
            self._state = ambient_state_for(
                self.vibe,
                self.key,
                elapsed_seconds=self.elapsed_seconds,
                enable_arpeggio=self.enable_arpeggio,
                enable_texture=self.enable_texture,
            )
            if self.running:
                self._apply_state()

    def reset_key(self, vibe: VibeProfile | None = None) -> None:
        with self._lock:
            if vibe is not None:
                self.vibe = vibe
            self.key = pick_modal_key(self.vibe)
            self._start_time = time.monotonic()
            self._state = ambient_state_for(
                self.vibe,
                self.key,
                enable_arpeggio=self.enable_arpeggio,
                enable_texture=self.enable_texture,
            )
            if self.running:
                self._apply_state()
            self._last_message = f"Key reset to {self.key.label}."

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time <= 0:
            return 0.0
        return time.monotonic() - self._start_time

    def status(self) -> PyoEngineStatus:
        with self._lock:
            return PyoEngineStatus(
                running=self.running,
                key=self._state.key.label,
                chord=self._state.chord.name,
                roman=self._state.chord.roman,
                layers=self._state.layers,
                tempo_bpm=self._state.tempo_bpm,
                message=self._last_message,
            )

    def music_state(self) -> AmbientMusicState:
        with self._lock:
            return self._state

    def _build_graph(self) -> None:
        pyo = self._pyo
        if pyo is None:
            raise RuntimeError("pyo is not loaded")

        wobble_depth = pyo.Sig(0.05)
        wobble = pyo.Sine(freq=0.035, mul=wobble_depth)

        drone_freqs = [pyo.Sig(110.0) for _ in range(4)]
        drone_amp = pyo.Sig(0.04)
        drone = pyo.Sine(
            freq=[
                drone_freqs[0] + wobble,
                drone_freqs[1] - wobble * 0.7,
                drone_freqs[2] + wobble * 0.4,
                drone_freqs[3],
            ],
            mul=drone_amp,
        ).mix(2)

        pad_freqs = [pyo.Sig(220.0) for _ in range(5)]
        pad_amp = pyo.Sig(0.025)
        pad_cutoff = pyo.Sig(1400.0)
        pad_raw = pyo.Sine(
            freq=[
                pad_freqs[0] + wobble * 0.25,
                pad_freqs[1] - wobble * 0.2,
                pad_freqs[2],
                pad_freqs[3] + wobble * 0.15,
                pad_freqs[4],
            ],
            mul=pad_amp,
        ).mix(2)
        pad = pyo.ButLP(pad_raw, freq=pad_cutoff)

        arp_freq = pyo.Sig(440.0)
        arp_amp = pyo.Sig(0.0)
        arp_env = pyo.Fader(fadein=0.02, fadeout=1.4, dur=1.8, mul=arp_amp)
        arpeggio = pyo.Sine(freq=arp_freq, mul=arp_env).mix(2)

        texture_amp = pyo.Sig(0.0)
        texture_cutoff = pyo.Sig(5000.0)
        texture = pyo.ButHP(pyo.Noise(mul=texture_amp), freq=texture_cutoff).mix(2)

        reverb_mix = pyo.Sig(0.78)
        master_amp = pyo.Sig(self.master_gain)
        source = drone + pad + arpeggio + texture
        master = pyo.Freeverb(source, size=0.92, damp=0.62, bal=reverb_mix, mul=master_amp).out()

        self._objects = {
            "wobble_depth": wobble_depth,
            "drone_freqs": drone_freqs,
            "drone_amp": drone_amp,
            "pad_freqs": pad_freqs,
            "pad_amp": pad_amp,
            "pad_cutoff": pad_cutoff,
            "arp_freq": arp_freq,
            "arp_amp": arp_amp,
            "arp_env": arp_env,
            "texture_amp": texture_amp,
            "texture_cutoff": texture_cutoff,
            "reverb_mix": reverb_mix,
            "master_amp": master_amp,
            "nodes": [wobble, drone, pad_raw, pad, arpeggio, texture, master],
        }

    def _apply_state(self) -> None:
        if not self._objects:
            return

        state = self._state
        vibe = state.vibe
        chord = state.chord
        root = _register(chord.root_midi, low=30, high=46)

        drone_notes = (root, root + 7, root + 12, root + 19)
        for sig, note in zip(self._objects["drone_freqs"], drone_notes, strict=False):
            sig.value = _midi_to_hz(note)

        pad_base = 48 if vibe.brightness < 0.35 else 55
        pad_notes = [_register(pitch, low=pad_base, high=pad_base + 24) for pitch in chord.pitches]
        while len(pad_notes) < 5:
            pad_notes.append(pad_notes[-1] + 12 if pad_notes else root + 24)
        for sig, note in zip(self._objects["pad_freqs"], pad_notes[:5], strict=False):
            sig.value = _midi_to_hz(note)

        self._objects["drone_amp"].value = 0.028 + (1.0 - vibe.brightness) * 0.025
        self._objects["pad_amp"].value = 0.018 + vibe.color * 0.026
        self._objects["pad_cutoff"].value = 450 + vibe.brightness * 2500 + vibe.color * 1400
        self._objects["arp_amp"].value = 0.0 if "arpeggio" not in state.layers else 0.018 + vibe.energy * 0.045
        self._objects["texture_amp"].value = 0.0 if "texture" not in state.layers else 0.004 + vibe.color * 0.014
        self._objects["texture_cutoff"].value = 3000 + vibe.brightness * 5000
        self._objects["reverb_mix"].value = 0.74 + vibe.color * 0.18
        self._objects["wobble_depth"].value = 0.02 + min(1.0, self._last_motion / 20.0) * 0.35

        if self._pattern is not None:
            self._pattern.time = max(0.7, 4.0 - vibe.energy * 2.8)

    def _play_arpeggio_note(self) -> None:
        with self._lock:
            if not self.running or "arpeggio" not in self._state.layers:
                return
            probability = 0.12 + self.vibe.energy * 0.55
            if self._rng.random() > probability:
                return

            scale = modal_scale_pitches(self.key, low=60, high=88)
            chord_pcs = {pitch % 12 for pitch in self._state.chord.pitches}
            options = [pitch for pitch in scale if pitch % 12 in chord_pcs] or scale
            if not options:
                return

            note = self._rng.choice(options)
            self._objects["arp_freq"].value = _midi_to_hz(note)
            self._objects["arp_env"].play()


def _import_pyo() -> Any:
    try:
        return importlib.import_module("pyo")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyo is required for ambient synthesis. Install it with "
            "`pip install pyo` or use `pip install -r requirements-ambient.txt`."
        ) from exc


def _midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _register(note: int, *, low: int, high: int) -> int:
    while note < low:
        note += 12
    while note > high:
        note -= 12
    return note
