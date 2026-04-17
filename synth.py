"""
Camera Fluid Synth
==================
Captures video from Pi camera, extracts visual features,
maps them to musical parameters, and plays via FluidSynth.

Feature → Music Mapping:
  Brightness    → Velocity / energy (dark=soft, bright=loud)
  Contrast      → Note spread (low=close voicing, high=wide)
  Saturation    → Instrument selection (low=piano, mid=strings, high=synth pad)
  Dominant Hue  → Key / root note (12 hue bins → 12 keys)
  Edge Density  → Rhythm speed (smooth=slow arpeggios, busy=fast patterns)
  Contour Count → Number of simultaneous voices (few=solo, many=full chord)
  Direction     → Pitch register (horizontal=low, vertical=high)
  Motion        → Note density + percussion (still=sparse, moving=busy+drums)

Requires:
  sudo apt install fluidsynth fluid-soundfont-gm python3-opencv python3-picamera2
  pip install pyfluidsynth --break-system-packages
"""

import argparse
import json
import os
import queue
import struct
import cv2
import numpy as np
import fluidsynth
import threading
import time
from pathlib import Path
from picamera2 import Picamera2
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# ── Config ────────────────────────────────────────────────────

SOUNDFONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
AUDIO_DRIVER = "alsa"
AUDIO_DEVICE = "hw:2,0"
STREAM_PORT = 8080
FRAME_SIZE = (320, 240)
GENERATED_MUSIC_DIR = Path(os.getenv("GENERATED_MUSIC_DIR", "generated_music"))
MIDI_DIR = Path(os.getenv("MIDI_DIR", str(Path.home() / "midi"))).expanduser()
MIDI_MANIFEST_PATH = os.getenv("MIDI_MANIFEST_PATH", "")

# General MIDI instrument presets
INSTRUMENTS = {
    'piano':       0,   # Acoustic Grand Piano
    'epiano':      4,   # Electric Piano
    'vibes':       11,  # Vibraphone
    'organ':       19,  # Church Organ
    'guitar':      25,  # Acoustic Guitar (nylon)
    'strings':     48,  # String Ensemble 1
    'slow_str':    49,  # String Ensemble 2
    'synth_pad':   88,  # Pad 1 (new age)
    'warm_pad':    89,  # Pad 2 (warm)
    'choir':       52,  # Choir Aahs
    'atmosphere':  99,  # FX 4 (atmosphere)
    'bass':        32,  # Acoustic Bass
    'synth_bass':  38,  # Synth Bass 1
}

# Instrument tiers based on saturation
SAT_INSTRUMENTS = [
    (30,  'piano'),       # very low saturation → clean piano
    (60,  'epiano'),      # low → electric piano
    (90,  'vibes'),       # low-mid → vibraphone
    (120, 'strings'),     # mid → strings
    (150, 'choir'),       # mid-high → choir
    (180, 'synth_pad'),   # high → synth pad
    (210, 'warm_pad'),    # very high → warm pad
    (255, 'atmosphere'),  # max → atmosphere
]

# 12 keys mapped to hue bins
KEYS = {
    0:  60,  # C
    1:  61,  # C#
    2:  62,  # D
    3:  63,  # D#
    4:  64,  # E
    5:  65,  # F
    6:  66,  # F#
    7:  67,  # G
    8:  68,  # G#
    9:  69,  # A
    10: 70,  # A#
    11: 71,  # B
}

# Scale intervals (semitones from root)
SCALES = {
    'major':      [0, 2, 4, 5, 7, 9, 11],
    'minor':      [0, 2, 3, 5, 7, 8, 10],
    'dorian':     [0, 2, 3, 5, 7, 9, 10],
    'phrygian':   [0, 1, 3, 5, 7, 8, 10],
    'lydian':     [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'pentatonic': [0, 2, 4, 7, 9],
    'blues':      [0, 3, 5, 6, 7, 10],
    'harmonic':   [0, 2, 3, 5, 7, 8, 11],
}

# Map hue bins to scale moods
HUE_SCALES = {
    0: 'major',       # C  - bright
    1: 'phrygian',    # C# - dark/spanish
    2: 'dorian',      # D  - jazzy
    3: 'minor',       # D# - melancholic
    4: 'lydian',      # E  - dreamy
    5: 'major',       # F  - warm
    6: 'blues',       # F# - gritty
    7: 'mixolydian',  # G  - groovy
    8: 'harmonic',    # G# - exotic
    9: 'pentatonic',  # A  - open
    10: 'minor',      # A# - dark
    11: 'dorian',     # B  - soulful
}

# Drum notes (General MIDI channel 10)
DRUMS = {
    'kick':    36,
    'snare':   38,
    'hihat':   42,
    'ohihat':  46,
    'crash':   49,
    'ride':    51,
    'clap':    39,
    'tom_low': 45,
    'tom_hi':  50,
}


# ── FluidSynth Engine ────────────────────────────────────────

SAMPLE_RATE = 44100

class FluidSynthEngine:
    def __init__(self, browser_mode=False):
        self.browser_mode = browser_mode
        self.sample_rate = SAMPLE_RATE
        self.fs = fluidsynth.Synth(gain=0.6, samplerate=float(SAMPLE_RATE))
        if not browser_mode:
            self.fs.start(driver=AUDIO_DRIVER, device=AUDIO_DEVICE)
        self.sf = self.fs.sfload(SOUNDFONT_PATH)

        # Set up channels
        # Ch 0: melody/lead
        # Ch 1: chords/pad
        # Ch 2: bass
        # Ch 9: drums (GM standard)
        self.fs.program_select(0, self.sf, 0, INSTRUMENTS['piano'])
        self.fs.program_select(1, self.sf, 0, INSTRUMENTS['strings'])
        self.fs.program_select(2, self.sf, 0, INSTRUMENTS['bass'])
        # Channel 9 is drums by default in GM

        self.active_notes = {0: [], 1: [], 2: []}
        self.lock = threading.RLock()
        self.midi_player = None
        self.midi_path = None
        self.midi_playing = False
        self.current_params = {}
        self.running = True

    def set_instrument(self, channel, preset):
        with self.lock:
            self.fs.program_select(channel, self.sf, 0, preset)

    def notes_off(self, channel):
        """Turn off all active notes on a channel"""
        for note in self.active_notes.get(channel, []):
            self.fs.noteoff(channel, note)
        self.active_notes[channel] = []

    def play_note(self, channel, note, velocity, duration=0.3):
        """Play a single note with auto-off"""
        note = max(0, min(127, note))
        velocity = max(0, min(127, velocity))
        self.fs.noteon(channel, note, velocity)
        self.active_notes.setdefault(channel, []).append(note)

        def off():
            time.sleep(duration)
            self.fs.noteoff(channel, note)
            if note in self.active_notes.get(channel, []):
                self.active_notes[channel].remove(note)

        threading.Thread(target=off, daemon=True).start()

    def play_drum(self, note, velocity=100):
        self.fs.noteon(9, note, velocity)
        threading.Thread(
            target=lambda: (time.sleep(0.1), self.fs.noteoff(9, note)),
            daemon=True
        ).start()

    def all_notes_off(self):
        """Silence generated notes and any sustained MIDI notes."""
        with self.lock:
            for channel in range(16):
                self.fs.cc(channel, 123, 0)  # all notes off
                self.fs.cc(channel, 120, 0)  # all sound off
            for channel in list(self.active_notes):
                self.active_notes[channel] = []

    def is_midi_playing(self):
        with self.lock:
            return self.midi_playing

    def apply_midi_feature_controls(self, features):
        """Map camera features to global MIDI playback controls."""
        if not features:
            return

        brightness = float(features.get('brightness', 128.0))
        saturation = float(features.get('saturation', 80.0))
        edge_density = float(features.get('edge_density', 0.05))
        motion = float(features.get('motion', 0.0))

        volume = int(45 + (brightness / 255.0) * 75)
        expression = int(55 + min(1.0, (motion + edge_density * 120) / 35.0) * 72)
        reverb = int(max(20, min(110, 95 - saturation / 3)))
        chorus = int(max(10, min(95, saturation / 2)))

        with self.lock:
            for channel in range(16):
                self.fs.cc(channel, 7, max(0, min(127, volume)))
                self.fs.cc(channel, 11, max(0, min(127, expression)))
                self.fs.cc(channel, 91, max(0, min(127, reverb)))
                self.fs.cc(channel, 93, max(0, min(127, chorus)))

    def play_midi_file(self, midi_path, features=None):
        """Play a standard MIDI file through the current FluidSynth instance."""
        path = Path(midi_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MIDI file does not exist: {path}")
        if not hasattr(fluidsynth, 'Player'):
            raise RuntimeError("pyFluidSynth Player support is not available in this environment")

        with self.lock:
            self.stop_midi()
            self.all_notes_off()
            self.apply_midi_feature_controls(features)

            player = fluidsynth.Player(self.fs)
            if hasattr(player, 'add'):
                player.add(str(path))
            elif hasattr(player, 'add_file'):
                player.add_file(str(path))
            else:
                raise RuntimeError("pyFluidSynth Player has no MIDI add method")
            if not hasattr(player, 'play'):
                raise RuntimeError("pyFluidSynth Player has no play method")
            player.play()
            self.midi_player = player
            self.midi_path = path
            self.midi_playing = True

        threading.Thread(target=self._monitor_midi_player, args=(player,), daemon=True).start()
        return path

    def _monitor_midi_player(self, player):
        try:
            if hasattr(player, 'join'):
                player.join()
        finally:
            with self.lock:
                if self.midi_player is player:
                    self.midi_playing = False

    def stop_midi(self):
        with self.lock:
            player = self.midi_player
            self.midi_player = None
            self.midi_playing = False
            self.midi_path = None

        if player is not None:
            if hasattr(player, 'stop'):
                try:
                    player.stop()
                except Exception:
                    pass
            self.all_notes_off()

    def stop(self):
        self.running = False
        self.stop_midi()
        for ch in self.active_notes:
            self.notes_off(ch)
        self.fs.delete()


# ── Feature Extraction ────────────────────────────────────────

prev_frame_gray = None

def extract_features(frame):
    global prev_frame_gray
    frame = cv2.resize(frame, FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))

    hue_hist = cv2.calcHist([hsv[:, :, 0]], [0], None, [12], [0, 180])
    dominant_hue = int(np.argmax(hue_hist))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    sobel_x = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)))
    sobel_y = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)))
    total_energy = sobel_x + sobel_y + 0.001
    direction = sobel_x / total_energy

    motion = 0.0
    if prev_frame_gray is not None:
        diff = cv2.absdiff(prev_frame_gray, gray)
        motion = float(np.mean(diff))
    prev_frame_gray = gray.copy()

    features = {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'dominant_hue': dominant_hue,
        'edge_density': edge_density,
        'contour_count': contour_count,
        'direction': direction,
        'motion': motion,
    }
    aux = {
        'edges': edges,
        'contours': contours,
        'frame_small': frame,
    }
    return features, aux


# ── Music Mapper ──────────────────────────────────────────────

def get_instrument_for_saturation(saturation):
    """Pick instrument based on saturation level"""
    for threshold, name in SAT_INSTRUMENTS:
        if saturation < threshold:
            return name, INSTRUMENTS[name]
    return 'atmosphere', INSTRUMENTS['atmosphere']

def build_scale_notes(root_midi, scale_intervals, octave_range=2, register_shift=0):
    """Build a list of MIDI notes from root + scale across octaves"""
    notes = []
    base = root_midi - 12 + register_shift
    for octave in range(octave_range):
        for interval in scale_intervals:
            note = base + (octave * 12) + interval
            if 36 <= note <= 96:
                notes.append(note)
    return sorted(notes)

def map_features_to_music(features):
    """Convert visual features into musical parameters"""

    # Root note from dominant hue
    root_midi = KEYS[features['dominant_hue']]
    scale_name = HUE_SCALES[features['dominant_hue']]
    scale_intervals = SCALES[scale_name]

    # Velocity from brightness (40-120)
    velocity = int(40 + (features['brightness'] / 255.0) * 80)

    # Octave range from contrast (1-4)
    octave_range = max(1, min(4, int(features['contrast'] / 20.0) + 1))

    # Register shift from directional energy (-12 to +12)
    register_shift = int((features['direction'] - 0.5) * 24)

    # Build available notes
    scale_notes = build_scale_notes(root_midi, scale_intervals, octave_range, register_shift)

    # Number of chord voices from contour count (1-5)
    n_voices = max(1, min(5, features['contour_count'] // 8 + 1))

    # Rhythm interval from edge density (0.5s slow → 0.08s fast)
    edge = features['edge_density']
    rhythm_interval = max(0.08, 0.5 - edge * 1.5)

    # Note duration from brightness (short when bright, long when dark)
    note_duration = max(0.1, 0.8 - (features['brightness'] / 255.0) * 0.6)

    # Instrument from saturation
    inst_name, inst_preset = get_instrument_for_saturation(features['saturation'])

    # Bass note
    bass_note = root_midi - 12
    if bass_note < 36:
        bass_note += 12

    # Motion drives drum activity and note density
    motion = features['motion']
    use_drums = motion > 5.0
    drum_velocity = int(min(120, motion * 4))

    # Pick bass instrument based on saturation
    if features['saturation'] > 120:
        bass_preset = INSTRUMENTS['synth_bass']
    else:
        bass_preset = INSTRUMENTS['bass']

    return {
        'root_midi': root_midi,
        'scale_name': scale_name,
        'scale_notes': scale_notes,
        'velocity': velocity,
        'octave_range': octave_range,
        'register_shift': register_shift,
        'n_voices': n_voices,
        'rhythm_interval': rhythm_interval,
        'note_duration': note_duration,
        'inst_name': inst_name,
        'inst_preset': inst_preset,
        'bass_note': bass_note,
        'bass_preset': bass_preset,
        'use_drums': use_drums,
        'drum_velocity': drum_velocity,
        'motion': motion,
        'brightness': features['brightness'],
        'edge_density': features['edge_density'],
    }


# ── Music Player Loop ─────────────────────────────────────────

def music_loop(synth):
    """Continuously plays music based on current parameters"""
    beat_count = 0
    current_inst = -1
    current_bass_inst = -1

    while synth.running:
        if synth.is_midi_playing():
            time.sleep(0.1)
            continue

        params = synth.current_params
        if not params:
            time.sleep(0.1)
            continue

        # Update instruments if changed
        if params['inst_preset'] != current_inst:
            synth.set_instrument(0, params['inst_preset'])  # melody
            synth.set_instrument(1, params['inst_preset'])  # chords
            current_inst = params['inst_preset']

        if params['bass_preset'] != current_bass_inst:
            synth.set_instrument(2, params['bass_preset'])
            current_bass_inst = params['bass_preset']

        scale_notes = params['scale_notes']
        if not scale_notes:
            time.sleep(0.1)
            continue

        velocity = params['velocity']
        n_voices = params['n_voices']
        duration = params['note_duration']

        # ── Melody (channel 0) ──
        # Pick a note from the scale, tendency towards higher notes
        mel_idx = np.random.randint(len(scale_notes) // 2, len(scale_notes))
        mel_note = scale_notes[min(mel_idx, len(scale_notes) - 1)]
        synth.play_note(0, mel_note, velocity, duration)

        # ── Chord (channel 1) ──
        # Play every other beat for pads
        if beat_count % 2 == 0:
            # Build chord from scale
            chord_indices = np.linspace(0, len(scale_notes) - 1, n_voices, dtype=int)
            for idx in chord_indices:
                chord_vel = max(30, velocity - 30)  # softer than melody
                synth.play_note(1, scale_notes[idx], chord_vel, duration * 2)

        # ── Bass (channel 2) ──
        # Play on every 4th beat
        if beat_count % 4 == 0:
            synth.play_note(2, params['bass_note'], min(110, velocity + 10), duration * 2)
        elif beat_count % 4 == 2:
            # Walking bass - move up a fifth
            fifth = params['bass_note'] + 7
            synth.play_note(2, fifth, min(100, velocity), duration)

        # ── Drums (channel 9) ──
        if params['use_drums']:
            dv = params['drum_velocity']

            # Kick on 1 and 3
            if beat_count % 4 == 0 or beat_count % 4 == 2:
                synth.play_drum(DRUMS['kick'], dv)

            # Snare on 2 and 4
            if beat_count % 4 == 1 or beat_count % 4 == 3:
                synth.play_drum(DRUMS['snare'], max(40, dv - 20))

            # Hi-hat - more frequent with more motion
            if params['motion'] > 10:
                synth.play_drum(DRUMS['hihat'], max(30, dv - 40))
            elif beat_count % 2 == 0:
                synth.play_drum(DRUMS['hihat'], max(20, dv - 50))

            # Crash on section changes (every 16 beats)
            if beat_count % 16 == 0:
                synth.play_drum(DRUMS['crash'], dv)

        beat_count += 1
        time.sleep(params['rhythm_interval'])


# ── Audio Broadcaster (browser mode) ──────────────────────────

audio_broadcaster = None

class AudioBroadcaster:
    """Pulls PCM samples from FluidSynth in real time and fans them out
    to all connected HTTP clients via per-client queues."""

    def __init__(self, engine, chunk_frames=1024):
        self.engine = engine
        self.chunk_frames = chunk_frames
        self.clients = []
        self.lock = threading.Lock()
        self.running = True

    def subscribe(self):
        q = queue.Queue(maxsize=64)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def run(self):
        chunk_duration = self.chunk_frames / self.engine.sample_rate
        while self.running and self.engine.running:
            start = time.monotonic()
            samples = self.engine.fs.get_samples(self.chunk_frames)
            data = fluidsynth.raw_audio_string(samples)
            with self.lock:
                for q in list(self.clients):
                    try:
                        q.put_nowait(data)
                    except queue.Full:
                        pass
            delay = chunk_duration - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)


def wav_stream_header(sample_rate=SAMPLE_RATE, channels=2, bits=16):
    """WAV header with indeterminate length for open-ended streaming."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b'RIFF' + struct.pack('<I', 0xFFFFFFFF) + b'WAVE' +
        b'fmt ' + struct.pack('<IHHIIHH', 16, 1, channels, sample_rate,
                              byte_rate, block_align, bits) +
        b'data' + struct.pack('<I', 0xFFFFFFFF)
    )


HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Camera Fluid Synth</title>
<style>
body { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:20px; margin:0; }
h1 { margin:0 0 16px; font-weight:300; letter-spacing:2px; }
img { max-width:90%; border:2px solid #0f8; border-radius:4px; }
audio { width:90%; max-width:640px; margin-top:16px; }
button { background:#0f8; border:0; border-radius:4px; color:#041; cursor:pointer; font-weight:700; margin:16px 8px 0; padding:10px 14px; }
button:disabled { background:#465; color:#abc; cursor:wait; }
pre { text-align:left; display:inline-block; font-size:12px; color:#8f8; max-width:90%; white-space:pre-wrap; }
.status { color:#9cf; font-size:14px; margin-top:10px; min-height:20px; }
</style></head>
<body>
<h1>CAMERA FLUID SYNTH</h1>
<img src="/video" alt="camera stream"><br>
<audio controls autoplay src="/audio">Audio streaming not supported in this browser.</audio>
<br>
<button id="generate">Generate music from camera</button>
<button id="playMidi">Play matching MIDI</button>
<button id="stopMidi">Stop MIDI</button>
<div class="status" id="musicStatus"></div>
<audio id="generatedAudio" controls></audio>
<pre id="elevenlabs"></pre>
<div class="status" id="midiStatus"></div>
<pre id="midi"></pre>
<pre id="feat"></pre>
<script>
async function tick() {
  try {
    const r = await fetch('/features');
    document.getElementById('feat').innerText = JSON.stringify(await r.json(), null, 2);
  } catch(e) {}
}
setInterval(tick, 500);

async function midiStatus() {
  try {
    const r = await fetch('/midi-status');
    const data = await r.json();
    document.getElementById('playMidi').disabled = data.state === 'running';
    document.getElementById('midiStatus').innerText = data.message || data.state;
    document.getElementById('midi').innerText = JSON.stringify({
      file: data.filename,
      mood: data.mood,
      reason: data.reason,
      description: data.description
    }, null, 2);
  } catch(e) {}
}

async function musicStatus() {
  try {
    const r = await fetch('/music-status');
    const data = await r.json();
    const button = document.getElementById('generate');
    button.disabled = data.state === 'running';
    document.getElementById('musicStatus').innerText = data.message || data.state;
    document.getElementById('elevenlabs').innerText = JSON.stringify({
      description: data.description,
      prompt: data.prompt,
      song_id: data.song_id,
      model: data.model
    }, null, 2);
    if (data.audio_url) {
      const audio = document.getElementById('generatedAudio');
      if (audio.src.indexOf(data.audio_url) === -1) {
        audio.src = data.audio_url;
      }
    }
  } catch(e) {}
}

document.getElementById('generate').addEventListener('click', async () => {
  const button = document.getElementById('generate');
  button.disabled = true;
  document.getElementById('musicStatus').innerText = 'Starting generation...';
  try {
    await fetch('/generate-music', {method: 'POST'});
  } catch(e) {
    document.getElementById('musicStatus').innerText = String(e);
  }
  musicStatus();
});

document.getElementById('playMidi').addEventListener('click', async () => {
  const button = document.getElementById('playMidi');
  button.disabled = true;
  document.getElementById('midiStatus').innerText = 'Choosing MIDI...';
  try {
    await fetch('/play-midi', {method: 'POST'});
  } catch(e) {
    document.getElementById('midiStatus').innerText = String(e);
  }
  midiStatus();
});

document.getElementById('stopMidi').addEventListener('click', async () => {
  try {
    await fetch('/stop-midi', {method: 'POST'});
  } catch(e) {
    document.getElementById('midiStatus').innerText = String(e);
  }
  midiStatus();
});

setInterval(musicStatus, 1500);
setInterval(midiStatus, 1000);
musicStatus();
midiStatus();
</script>
</body></html>"""


# ── Video Stream ──────────────────────────────────────────────

latest_frame = None
latest_camera_frame = None
latest_features = {}
latest_music = {}
synth_engine = None
frame_lock = threading.Lock()
generation_lock = threading.Lock()
midi_lock = threading.Lock()
generated_music_path = None
generated_music_mime = "audio/mpeg"
generation_status = {
    "state": "idle",
    "message": "Press Generate music from camera to create an ElevenLabs track.",
    "description": "",
    "prompt": "",
    "song_id": "",
    "model": "",
    "audio_url": "",
    "updated_at": "",
}
midi_status = {
    "state": "idle",
    "message": f"Put .mid files in {MIDI_DIR} and press Play matching MIDI.",
    "filename": "",
    "path": "",
    "reason": "",
    "description": "",
    "mood": "",
    "updated_at": "",
}


def update_generation_status(**fields):
    with generation_lock:
        generation_status.update(fields)
        generation_status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def get_generation_status():
    with generation_lock:
        return dict(generation_status)


def update_midi_status(**fields):
    with midi_lock:
        midi_status.update(fields)
        midi_status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def get_midi_status():
    with midi_lock:
        return dict(midi_status)


def generate_camera_music(snapshot, feature_snapshot):
    global generated_music_path, generated_music_mime

    try:
        update_generation_status(
            state="running",
            message="Describing the camera image with OpenAI...",
            description="",
            prompt="",
            song_id="",
            model="",
            audio_url="",
        )

        from module import (
            build_elevenlabs_prompt,
            describe_image,
            generate_elevenlabs_music,
            save_elevenlabs_music_result,
        )

        description_result = describe_image(snapshot, include_parts=True, max_parts=6)
        prompt = build_elevenlabs_prompt(
            description_result.description,
            features=description_result.features or feature_snapshot,
        )
        update_generation_status(
            message="Generating music with ElevenLabs...",
            description=description_result.description,
            prompt=prompt,
        )

        result = generate_elevenlabs_music(prompt)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        saved = save_elevenlabs_music_result(result, GENERATED_MUSIC_DIR, stem=f"camera_{timestamp}")
        audio_path = saved.get("audio")
        if audio_path is None:
            raise RuntimeError("ElevenLabs response did not include audio data")

        with generation_lock:
            generated_music_path = audio_path
            generated_music_mime = result.mime_type

        update_generation_status(
            state="complete",
            message=f"Generated {audio_path.name}",
            song_id=result.song_id or "",
            model=result.model_id,
            audio_url=f"/generated-music?t={int(time.time())}",
        )
    except Exception as exc:
        update_generation_status(
            state="error",
            message=f"Music generation failed: {exc}",
        )


def select_and_play_camera_midi(snapshot, feature_snapshot):
    try:
        if synth_engine is None:
            raise RuntimeError("FluidSynth engine is not ready")

        update_midi_status(
            state="running",
            message="Describing image and choosing a MIDI file...",
            filename="",
            path="",
            reason="",
            description="",
            mood="",
        )

        from module import choose_midi_for_image

        manifest_path = MIDI_MANIFEST_PATH or None
        selection = choose_midi_for_image(
            snapshot,
            MIDI_DIR,
            features=feature_snapshot,
            manifest_path=manifest_path,
        )
        played_path = synth_engine.play_midi_file(selection.path, features=feature_snapshot)
        update_midi_status(
            state="playing",
            message=f"Playing {selection.filename}",
            filename=selection.filename,
            path=str(played_path),
            reason=selection.reason,
            description=selection.image_description,
            mood=selection.mood,
        )
    except Exception as exc:
        update_midi_status(
            state="error",
            message=f"MIDI selection/playback failed: {exc}",
        )


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/features':
            return self._serve_features()
        if self.path == '/music-status':
            return self._serve_music_status()
        if self.path == '/midi-status':
            return self._serve_midi_status()
        if self.path.startswith('/generated-music'):
            return self._serve_generated_music()
        if self.path == '/audio' and audio_broadcaster is not None:
            return self._serve_audio()
        if self.path == '/' and audio_broadcaster is not None:
            return self._serve_html()
        # '/' (non-browser mode) and '/video' both serve the MJPEG stream
        return self._serve_video()

    def do_POST(self):
        if self.path == '/generate-music':
            return self._start_music_generation()
        if self.path == '/play-midi':
            return self._start_midi_selection()
        if self.path == '/stop-midi':
            return self._stop_midi()
        self.send_error(404)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_features(self):
        with frame_lock:
            data = {
                'features': {k: round(v, 2) if isinstance(v, float) else v
                             for k, v in latest_features.items()},
                'music': {k: v for k, v in latest_music.items()
                          if k not in ('scale_notes',)},
            }
        self._send_json(data)

    def _serve_music_status(self):
        self._send_json(get_generation_status())

    def _serve_midi_status(self):
        status = get_midi_status()
        if synth_engine is not None and status.get("state") == "playing" and not synth_engine.is_midi_playing():
            update_midi_status(state="idle", message="MIDI playback finished.")
            status = get_midi_status()
        self._send_json(status)

    def _start_music_generation(self):
        with generation_lock:
            if generation_status["state"] == "running":
                return self._send_json(get_generation_status(), status=409)

        with frame_lock:
            if latest_camera_frame is None:
                return self._send_json({"state": "error", "message": "No camera frame is available yet"}, status=503)
            snapshot = latest_camera_frame.copy()
            feature_snapshot = dict(latest_features)

        update_generation_status(state="running", message="Queued music generation...")
        thread = threading.Thread(
            target=generate_camera_music,
            args=(snapshot, feature_snapshot),
            daemon=True,
        )
        thread.start()
        self._send_json(get_generation_status(), status=202)

    def _start_midi_selection(self):
        with midi_lock:
            if midi_status["state"] == "running":
                return self._send_json(get_midi_status(), status=409)

        with frame_lock:
            if latest_camera_frame is None:
                return self._send_json({"state": "error", "message": "No camera frame is available yet"}, status=503)
            snapshot = latest_camera_frame.copy()
            feature_snapshot = dict(latest_features)

        update_midi_status(state="running", message="Queued MIDI selection...")
        thread = threading.Thread(
            target=select_and_play_camera_midi,
            args=(snapshot, feature_snapshot),
            daemon=True,
        )
        thread.start()
        self._send_json(get_midi_status(), status=202)

    def _stop_midi(self):
        if synth_engine is not None:
            synth_engine.stop_midi()
        update_midi_status(state="idle", message="MIDI playback stopped.")
        self._send_json(get_midi_status())

    def _serve_generated_music(self):
        with generation_lock:
            path = generated_music_path
            mime_type = generated_music_mime

        if path is None or not Path(path).exists():
            self.send_error(404, "No generated music yet")
            return

        data = Path(path).read_bytes()
        self.send_response(200)
        self.send_header('Content-type', mime_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache, no-store')
        self.end_headers()
        self.wfile.write(data)

    def _serve_html(self):
        body = HTML_PAGE.encode()
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_audio(self):
        self.send_response(200)
        self.send_header('Content-type', 'audio/wav')
        self.send_header('Cache-Control', 'no-cache, no-store')
        self.end_headers()
        q = audio_broadcaster.subscribe()
        try:
            self.wfile.write(wav_stream_header())
            while True:
                try:
                    data = q.get(timeout=5)
                except queue.Empty:
                    continue
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            audio_broadcaster.unsubscribe(q)

    def _serve_video(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while True:
            try:
                with frame_lock:
                    if latest_frame is None:
                        continue
                    frame = latest_frame.copy()
                _, jpeg = cv2.imencode('.jpg', frame)
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                time.sleep(0.05)
            except BrokenPipeError:
                break

    def log_message(self, format, *args):
        pass


# ── Feature Visualization ─────────────────────────────────────

VIDEO_W, VIDEO_H = FRAME_SIZE           # 320 x 240
PANEL_W = 320
CANVAS_W = VIDEO_W + PANEL_W            # 640
CANVAS_H = 480

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, pos, scale=0.4, color=(220, 220, 220), thick=1):
    cv2.putText(img, s, pos, FONT, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, fg=(0, 200, 120), bg=(40, 40, 48)):
    frac = max(0.0, min(1.0, float(frac)))
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = int(w * frac)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 100), 1)


def render_visualization(frame, features, music, aux):
    """Composite dashboard: video + edge preview + feature→audio panel."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 20, dtype=np.uint8)

    # ── Top-left: live video with contour + direction overlays ──
    video = cv2.resize(frame, (VIDEO_W, VIDEO_H)).copy()
    if aux.get('contours') is not None:
        cv2.drawContours(video, aux['contours'], -1, (0, 255, 180), 1)

    cx, cy = VIDEO_W // 2, VIDEO_H // 2
    d = features['direction']  # 1.0 = horizontal edges dominate, 0.0 = vertical
    h_len = int(d * 70)
    v_len = int((1.0 - d) * 70)
    cv2.arrowedLine(video, (cx - h_len, cy), (cx + h_len, cy),
                    (80, 180, 255), 2, tipLength=0.25)
    cv2.arrowedLine(video, (cx, cy - v_len), (cx, cy + v_len),
                    (255, 180, 80), 2, tipLength=0.25)
    _text(video, "contours / direction", (6, 14), scale=0.35, color=(0, 255, 180))
    canvas[0:VIDEO_H, 0:VIDEO_W] = video

    # ── Bottom-left: edge preview (heatmap) ──
    edges = aux.get('edges')
    if edges is not None:
        edge_vis = cv2.applyColorMap(edges, cv2.COLORMAP_INFERNO)
        edge_vis = cv2.resize(edge_vis, (VIDEO_W, CANVAS_H - VIDEO_H))
        canvas[VIDEO_H:CANVAS_H, 0:VIDEO_W] = edge_vis
    _text(canvas, f"EDGES  density {features['edge_density']:.2f}",
          (8, VIDEO_H + 18), scale=0.42, color=(255, 230, 120), thick=1)
    _text(canvas, f"-> rhythm {music['rhythm_interval']:.2f}s/beat",
          (8, VIDEO_H + 36), scale=0.38, color=(120, 180, 255))

    # ── Right panel: feature → audio ──
    px, py = VIDEO_W + 12, 18
    _text(canvas, "FEATURE  ->  AUDIO", (px, py), scale=0.5, color=(200, 230, 255), thick=1)
    py += 14

    # Hue palette row
    swatch_w = (PANEL_W - 24) // 12
    dom = features['dominant_hue']
    for i in range(12):
        hue_deg = int(i * 15 + 7)
        hsv = np.uint8([[[hue_deg, 220, 235]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        sx = px + i * swatch_w
        cv2.rectangle(canvas, (sx, py), (sx + swatch_w - 2, py + 18),
                      (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
    sx = px + dom * swatch_w
    cv2.rectangle(canvas, (sx - 2, py - 2), (sx + swatch_w - 1, py + 20),
                  (255, 255, 255), 2)
    py += 24
    _text(canvas,
          f"HUE bin {dom}  ->  KEY {KEY_NAMES[dom]}  {music['scale_name']}",
          (px, py + 10), scale=0.4, color=(150, 240, 200))
    py += 22

    # Generic feature row
    def row(label, frac, raw, arrow, bar_color=(0, 200, 120)):
        nonlocal py
        _text(canvas, label, (px, py + 12), scale=0.42, color=(200, 200, 210))
        _bar(canvas, px + 52, py + 3, 148, 14, frac, fg=bar_color)
        _text(canvas, raw, (px + 206, py + 14), scale=0.36, color=(230, 230, 230))
        _text(canvas, arrow, (px + 6, py + 30), scale=0.38, color=(120, 190, 255))
        py += 40

    row('BRT',
        features['brightness'] / 255.0,
        f"{features['brightness']:.0f}",
        f"-> velocity {music['velocity']}",
        bar_color=(int(features['brightness']), int(features['brightness']), int(features['brightness'])))

    row('CTR',
        min(1.0, features['contrast'] / 80.0),
        f"{features['contrast']:.0f}",
        f"-> octave range {music['octave_range']}",
        bar_color=(120, 220, 240))

    # Saturation row uses the dominant hue for an illustrative fill color
    hsv = np.uint8([[[int(dom * 15 + 7),
                      int(min(255, features['saturation'])), 230]]])
    sat_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    row('SAT',
        features['saturation'] / 255.0,
        f"{features['saturation']:.0f}",
        f"-> instrument {music['inst_name']}",
        bar_color=(int(sat_bgr[0]), int(sat_bgr[1]), int(sat_bgr[2])))

    row('CNT',
        min(1.0, features['contour_count'] / 40.0),
        f"{features['contour_count']}",
        f"-> voices {music['n_voices']}",
        bar_color=(120, 240, 180))

    row('DIR',
        features['direction'],
        f"{features['direction']:.2f}",
        f"-> register shift {music['register_shift']:+d}",
        bar_color=(100, 160, 255))

    row('MOT',
        min(1.0, features['motion'] / 30.0),
        f"{features['motion']:.1f}",
        f"-> drums {'ON' if music['use_drums'] else 'off'}  vel {music['drum_velocity']}",
        bar_color=(255, 120, 100))

    # Drum flash indicator
    if music['use_drums']:
        cv2.circle(canvas, (CANVAS_W - 20, py - 20), 8, (80, 255, 120), -1)

    return canvas


# ── Main ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Camera Fluid Synth")
    parser.add_argument('--browser', action='store_true',
                        help='Stream audio to the browser instead of the Pi audio jack')
    parser.add_argument('--midi-dir', default=str(MIDI_DIR),
                        help='Folder of .mid/.midi files for OpenAI-selected playback')
    parser.add_argument('--midi-manifest', default=MIDI_MANIFEST_PATH,
                        help='Optional path to midi_manifest.json metadata')
    return parser.parse_args()


def main():
    global latest_frame, latest_camera_frame, latest_features, latest_music, audio_broadcaster
    global synth_engine, MIDI_DIR, MIDI_MANIFEST_PATH

    args = parse_args()
    MIDI_DIR = Path(args.midi_dir).expanduser()
    MIDI_MANIFEST_PATH = str(Path(args.midi_manifest).expanduser()) if args.midi_manifest else ""
    update_midi_status(message=f"Put .mid files in {MIDI_DIR} and press Play matching MIDI.")

    print("Starting Camera Fluid Synth...")
    print("=" * 50)

    # Init camera
    camera = Picamera2()
    camera.start()
    print("[OK] Camera started")

    # Init synth
    synth = FluidSynthEngine(browser_mode=args.browser)
    synth_engine = synth
    print(f"[OK] FluidSynth started (output: {'browser' if args.browser else 'alsa ' + AUDIO_DEVICE})")

    # In browser mode, render samples and fan them out to HTTP clients
    if args.browser:
        audio_broadcaster = AudioBroadcaster(synth)
        broadcaster_thread = threading.Thread(target=audio_broadcaster.run, daemon=True)
        broadcaster_thread.start()
        print("[OK] Audio broadcaster started")

    # Start music thread
    music_thread = threading.Thread(target=music_loop, args=(synth,), daemon=True)
    music_thread.start()
    print("[OK] Music loop started")

    # Start HTTP stream
    http_server = ThreadingHTTPServer(('0.0.0.0', STREAM_PORT), StreamHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    if args.browser:
        print(f"[OK] Open http://<pi-ip>:{STREAM_PORT}/ for video + audio")
        print(f"[OK] Audio WAV stream at /audio, video at /video")
    else:
        print(f"[OK] Video stream at http://<pi-ip>:{STREAM_PORT}")
    print(f"[OK] MIDI library: {MIDI_DIR}")
    print(f"[OK] MIDI manifest: {MIDI_MANIFEST_PATH or MIDI_DIR / 'midi_manifest.json'}")
    print(f"[OK] Features JSON at http://<pi-ip>:{STREAM_PORT}/features")
    print("=" * 50)
    print("Point the camera around! Ctrl+C to stop.\n")

    try:
        while True:
            raw = camera.capture_array()
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

            features, aux = extract_features(frame)
            music_params = map_features_to_music(features)

            # Update synth params
            synth.current_params = music_params

            # Build dashboard-style visualization
            display_frame = render_visualization(frame, features, music_params, aux)

            with frame_lock:
                latest_frame = display_frame
                latest_camera_frame = frame.copy()
                latest_features = features
                latest_music = music_params

            # Print status
            print(
                f"Key: {music_params['scale_name']:>10} | "
                f"Inst: {music_params['inst_name']:>12} | "
                f"Vel: {music_params['velocity']:>3} | "
                f"Voices: {music_params['n_voices']} | "
                f"Rhythm: {music_params['rhythm_interval']:.2f}s | "
                f"Drums: {'ON ' if music_params['use_drums'] else 'OFF'}",
                end='\r'
            )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        synth.stop()
        camera.stop()
        http_server.shutdown()
        print("Done.")


if __name__ == '__main__':
    main()
