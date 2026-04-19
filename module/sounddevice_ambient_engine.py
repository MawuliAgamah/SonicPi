"""NumPy + sounddevice ambient engine for camera-driven synthesis."""

from __future__ import annotations

import importlib
import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ambient_model import (
    AmbientMusicState,
    ModalKey,
    ambient_state_for,
    modal_scale_pitches,
    pick_modal_key,
)
from .midi_arranger import VibeProfile


@dataclass(frozen=True)
class SoundDeviceEngineStatus:
    running: bool
    key: str
    chord: str
    roman: str
    layers: tuple[str, ...]
    tempo_bpm: int
    message: str


class SoundDeviceAmbientEngine:
    """A simple real-time synth using a sounddevice OutputStream callback."""

    def __init__(
        self,
        *,
        initial_vibe: VibeProfile | None = None,
        key: ModalKey | None = None,
        enable_arpeggio: bool = True,
        enable_texture: bool = True,
        master_gain: float = 0.45,
        sample_rate: int = 44100,
        blocksize: int = 512,
        audio_device: str | int | None = None,
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
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        self.audio_device = audio_device

        self._stream: Any | None = None
        self._state = ambient_state_for(
            self.vibe,
            self.key,
            enable_arpeggio=enable_arpeggio,
            enable_texture=enable_texture,
        )
        self._start_time = 0.0
        self._last_motion = 0.0
        self._last_message = "Stopped."
        self._lock = threading.RLock()
        self._rng = random.Random()
        self._noise_rng = np.random.default_rng()

        self._drone_phase = np.zeros(4, dtype=np.float64)
        self._pad_phase = np.zeros(6, dtype=np.float64)
        self._arp_phase = 0.0
        self._lfo_phase = 0.0
        self._texture_phase = 0.0
        self._arp_timer = 0
        self._arp_remaining = 0
        self._arp_duration = 1
        self._arp_freq = 440.0
        self._delay_buffer = np.zeros((max(1, int(self.sample_rate * 1.25)), 2), dtype=np.float32)
        self._delay_index = 0

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        with self._lock:
            if self.running:
                return

            sounddevice = _import_sounddevice()
            self._start_time = time.monotonic()
            self._stream = sounddevice.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=2,
                dtype="float32",
                device=self.audio_device,
                callback=self._callback,
            )
            self._stream.start()
            self._last_message = "Ambient engine running with sounddevice."

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
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
            self._last_message = f"Key reset to {self.key.label}."

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time <= 0:
            return 0.0
        return time.monotonic() - self._start_time

    def status(self) -> SoundDeviceEngineStatus:
        with self._lock:
            return SoundDeviceEngineStatus(
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

    def _callback(self, outdata, frames: int, time_info, status) -> None:
        if status:
            self._last_message = str(status)
        try:
            outdata[:] = self._render(frames)
        except Exception as exc:
            self._last_message = f"Audio callback failed: {exc}"
            outdata.fill(0)

    def _render(self, frames: int) -> np.ndarray:
        with self._lock:
            state = self._state
            vibe = self.vibe
            motion = self._last_motion

        root = _register(state.chord.root_midi, low=30, high=46)
        drone_notes = (root, root + 7, root + 12, root + 19)
        drone_freqs = np.array([_midi_to_hz(note) for note in drone_notes], dtype=np.float64)

        pad_base = 48 if vibe.brightness < 0.35 else 55
        pad_notes = [_register(pitch, low=pad_base, high=pad_base + 24) for pitch in state.chord.pitches]
        while len(pad_notes) < 6:
            pad_notes.append(pad_notes[-1] + 12 if pad_notes else root + 24)
        pad_freqs = np.array([_midi_to_hz(note) for note in pad_notes[:6]], dtype=np.float64)

        lfo = self._sine_lfo(frames, 0.035)
        wobble_hz = (0.02 + min(1.0, motion / 20.0) * 0.32) * lfo

        drone = self._osc_bank(
            drone_freqs + wobble_hz[-1] * np.array([1.0, -0.7, 0.4, 0.2]),
            self._drone_phase,
            frames,
        )
        drone_amp = 0.035 + (1.0 - vibe.brightness) * 0.025
        drone *= drone_amp

        pad = self._osc_bank(pad_freqs, self._pad_phase, frames)
        tremolo = 0.72 + 0.28 * self._sine_lfo(frames, 0.018)
        pad_amp = 0.018 + vibe.color * 0.032
        pad *= pad_amp * tremolo

        signal = drone + pad

        if "arpeggio" in state.layers:
            signal += self._render_arpeggio(frames, state, vibe)

        if "texture" in state.layers:
            signal += self._render_texture(frames, vibe)

        stereo = np.column_stack((signal * 0.96, signal * 0.86))
        stereo = self._delay_wash(stereo, vibe)
        stereo *= self.master_gain
        return np.tanh(stereo * 1.4).astype(np.float32)

    def _osc_bank(self, freqs: np.ndarray, phases: np.ndarray, frames: int) -> np.ndarray:
        sample_index = np.arange(frames, dtype=np.float64)
        signal = np.zeros(frames, dtype=np.float64)
        for index, freq in enumerate(freqs):
            inc = _TAU * float(freq) / self.sample_rate
            phase = phases[index]
            signal += np.sin(phase + inc * sample_index)
            phases[index] = (phase + inc * frames) % _TAU
        return signal / max(1, len(freqs))

    def _sine_lfo(self, frames: int, freq: float) -> np.ndarray:
        sample_index = np.arange(frames, dtype=np.float64)
        inc = _TAU * freq / self.sample_rate
        values = np.sin(self._lfo_phase + inc * sample_index)
        self._lfo_phase = (self._lfo_phase + inc * frames) % _TAU
        return values

    def _render_arpeggio(
        self,
        frames: int,
        state: AmbientMusicState,
        vibe: VibeProfile,
    ) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float64)
        cursor = 0
        while cursor < frames:
            if self._arp_remaining <= 0 and self._arp_timer <= 0:
                probability = 0.12 + vibe.energy * 0.55
                if self._rng.random() < probability:
                    scale = modal_scale_pitches(state.key, low=60, high=88)
                    chord_pcs = {pitch % 12 for pitch in state.chord.pitches}
                    options = [pitch for pitch in scale if pitch % 12 in chord_pcs] or scale
                    if options:
                        self._arp_freq = _midi_to_hz(self._rng.choice(options))
                        self._arp_duration = max(1, int(self.sample_rate * (1.4 + (1.0 - vibe.energy) * 1.0)))
                        self._arp_remaining = self._arp_duration
                wait_seconds = max(0.75, 4.0 - vibe.energy * 2.8)
                self._arp_timer = int(self.sample_rate * wait_seconds)

            if self._arp_remaining > 0:
                count = min(frames - cursor, self._arp_remaining)
                local = np.arange(count, dtype=np.float64)
                inc = _TAU * self._arp_freq / self.sample_rate
                start = self._arp_phase
                wave = np.sin(start + inc * local)
                self._arp_phase = (start + inc * count) % _TAU
                progress = 1.0 - (self._arp_remaining - local) / max(1, self._arp_duration)
                attack = np.clip(progress / 0.08, 0.0, 1.0)
                release = np.clip((self._arp_remaining - local) / (self.sample_rate * 0.9), 0.0, 1.0)
                env = np.minimum(attack, release)
                out[cursor : cursor + count] += wave * env * (0.02 + vibe.energy * 0.06)
                self._arp_remaining -= count
                cursor += count
            else:
                count = min(frames - cursor, self._arp_timer)
                self._arp_timer -= count
                cursor += max(1, count)
        return out

    def _render_texture(self, frames: int, vibe: VibeProfile) -> np.ndarray:
        noise = self._noise_rng.normal(0.0, 1.0, frames)
        gate_freq = 0.35 + vibe.energy * 1.2
        gate = (self._sine_texture_lfo(frames, gate_freq) > (0.55 - vibe.color * 0.35)).astype(np.float64)
        amp = 0.002 + vibe.color * 0.018
        return noise * gate * amp

    def _sine_texture_lfo(self, frames: int, freq: float) -> np.ndarray:
        sample_index = np.arange(frames, dtype=np.float64)
        inc = _TAU * freq / self.sample_rate
        values = np.sin(self._texture_phase + inc * sample_index)
        self._texture_phase = (self._texture_phase + inc * frames) % _TAU
        return values

    def _delay_wash(self, stereo: np.ndarray, vibe: VibeProfile) -> np.ndarray:
        wet = 0.22 + vibe.color * 0.24
        feedback = 0.22 + vibe.brightness * 0.18
        delay_samples = max(1, int(self.sample_rate * (0.42 + (1.0 - vibe.energy) * 0.32)))
        out = np.empty_like(stereo)
        size = len(self._delay_buffer)
        read_offset = delay_samples % size
        for index in range(len(stereo)):
            read_index = (self._delay_index - read_offset) % size
            delayed = self._delay_buffer[read_index]
            sample = stereo[index] + delayed * wet
            self._delay_buffer[self._delay_index] = stereo[index] + delayed * feedback
            out[index] = sample
            self._delay_index = (self._delay_index + 1) % size
        return out


def _import_sounddevice() -> Any:
    try:
        return importlib.import_module("sounddevice")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sounddevice is required for the default ambient engine. Install it with "
            "`pip install sounddevice` or `pip install -r requirements-ambient.txt`."
        ) from exc


def _midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _register(note: int, *, low: int, high: int) -> int:
    while note < low:
        note += 12
    while note > high:
        note -= 12
    return note


_TAU = math.tau
