"""Browser-based Mac webcam test runner for camera-to-MIDI selection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_MIDI_DIR = Path("midi/free_chord_progressions")
DEFAULT_DRUM_DIR = Path("midi/dmp_drum_patterns")
DEFAULT_ARRANGEMENT_DIR = Path("generated_arrangements")
DEFAULT_PORT = 8090
FRAME_INTERVAL = 0.05
DEFAULT_FLUIDSYNTH_AUDIO_DRIVER = "coreaudio" if sys.platform == "darwin" else "alsa"
COMMON_SOUNDFONT_PATHS = [
    "/opt/homebrew/share/fluid-soundfont/FluidR3_GM.sf2",
    "/usr/local/share/fluid-soundfont/FluidR3_GM.sf2",
    "/opt/homebrew/share/soundfonts/FluidR3_GM.sf2",
    "/usr/local/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
]


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Mac Camera MIDI Test</title>
<style>
body { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:20px; margin:0; }
h1 { margin:0 0 16px; font-weight:300; letter-spacing:1px; }
img { width:90%; max-width:760px; border:2px solid #0f8; border-radius:4px; background:#000; }
button { background:#0f8; border:0; border-radius:4px; color:#041; cursor:pointer; font-weight:700; margin:16px 6px 0; padding:10px 14px; }
button:disabled { background:#465; color:#abc; cursor:wait; }
.status { color:#9cf; font-size:14px; margin:12px auto; max-width:860px; min-height:20px; }
pre { color:#8f8; display:inline-block; font-size:12px; max-width:90%; text-align:left; white-space:pre-wrap; }
</style>
</head>
<body>
<h1>MAC CAMERA MIDI TEST</h1>
<img src="/video" alt="webcam stream"><br>
<button id="snapshot">Snapshot and choose MIDI</button>
<button id="arrange">Play chord + drums</button>
<button id="replay">Replay last MIDI</button>
<button id="stop">Stop MIDI</button>
<div class="status" id="status"></div>
<pre id="details"></pre>
<script>
async function refreshStatus() {
  try {
    const response = await fetch('/status');
    const data = await response.json();
    document.getElementById('snapshot').disabled = data.state === 'running';
    document.getElementById('status').innerText = data.message || data.state;
    document.getElementById('details').innerText = JSON.stringify({
      filename: data.filename,
      mood: data.mood,
      reason: data.reason,
      description: data.description,
      midi_dir: data.midi_dir,
      drum_dir: data.drum_dir,
      fluidsynth: data.fluidsynth,
      soundfont: data.soundfont,
      soundfont_log: data.soundfont_log,
      arrangement: data.arrangement
    }, null, 2);
  } catch(e) {}
}

async function post(path) {
  await fetch(path, {method: 'POST'});
  refreshStatus();
}

document.getElementById('snapshot').addEventListener('click', () => post('/snapshot'));
document.getElementById('arrange').addEventListener('click', () => post('/arrange'));
document.getElementById('replay').addEventListener('click', () => post('/replay'));
document.getElementById('stop').addEventListener('click', () => post('/stop'));
setInterval(refreshStatus, 1000);
refreshStatus();
</script>
</body>
</html>"""


class MidiPlayer:
    """Small local MIDI playback wrapper."""

    def __init__(
        self,
        soundfont: str | None = None,
        prefer_fluidsynth: bool = True,
        audio_driver: str = DEFAULT_FLUIDSYNTH_AUDIO_DRIVER,
        rendered_audio_dir: str | Path = "generated_arrangements/rendered_audio",
    ):
        if soundfont:
            self.soundfont = Path(soundfont).expanduser()
            self.soundfont_log = [f"--soundfont provided: {self.soundfont}"]
        else:
            self.soundfont, self.soundfont_log = detect_soundfont()
        self.prefer_fluidsynth = prefer_fluidsynth
        self.audio_driver = audio_driver
        self.rendered_audio_dir = Path(rendered_audio_dir).expanduser()
        self.process: subprocess.Popen[bytes] | None = None
        print("[soundfont] detection log:", flush=True)
        for line in self.soundfont_log:
            print(f"[soundfont] {line}", flush=True)

    def play(self, midi_path: str | Path, *, soundfont: str | Path | None = None) -> str:
        path = Path(midi_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MIDI file does not exist: {path}")

        active_soundfont = Path(soundfont).expanduser() if soundfont else self.soundfont

        self.stop()
        fluidsynth_bin = shutil.which("fluidsynth")
        print(f"[midi] selected file: {path}", flush=True)
        print(f"[fluidsynth] executable: {fluidsynth_bin or 'not found'}", flush=True)
        print(f"[fluidsynth] soundfont: {active_soundfont or 'not found'}", flush=True)
        if self.prefer_fluidsynth and fluidsynth_bin and active_soundfont and active_soundfont.exists():
            command = [fluidsynth_bin, "-a", self.audio_driver, str(active_soundfont), str(path)]
            print(f"[fluidsynth] command: {' '.join(command)}", flush=True)
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.5)
            if self.process.poll() not in (None, 0):
                _, stderr = self.process.communicate(timeout=1)
                self.process = None
                error = stderr.decode("utf-8", errors="replace").strip()
                print(f"[fluidsynth] realtime playback failed: {error or 'unknown error'}", flush=True)
                fallback = self._render_audio_fallback(
                    path,
                    active_soundfont=active_soundfont,
                    fluidsynth_bin=fluidsynth_bin,
                )
                return (
                    "FluidSynth could not open the live audio device, so it rendered "
                    f"and opened audio instead: {fallback.name}"
                )
            return f"Playing with FluidSynth ({active_soundfont.name}): {path.name}"

        subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if self.prefer_fluidsynth and fluidsynth_bin and not active_soundfont:
            return "Opened MIDI in macOS. FluidSynth is installed, but no SoundFont was found. Install fluid-soundfont or pass --soundfont PATH."
        if self.prefer_fluidsynth and fluidsynth_bin and active_soundfont and not active_soundfont.exists():
            return f"Opened MIDI in macOS. SoundFont path does not exist: {active_soundfont}"
        if self.prefer_fluidsynth and not fluidsynth_bin:
            return "Opened MIDI in macOS. Install fluidsynth to audition through the target synth."
        return f"Opened MIDI: {path.name}"

    def _render_audio_fallback(
        self,
        midi_path: Path,
        *,
        active_soundfont: Path,
        fluidsynth_bin: str,
    ) -> Path:
        """Render MIDI to WAV when realtime CoreAudio/PortAudio cannot start."""

        self.rendered_audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.rendered_audio_dir / f"{midi_path.stem}_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        command = [
            fluidsynth_bin,
            "-ni",
            "-q",
            "-F",
            str(audio_path),
            "-T",
            "wav",
            str(active_soundfont),
            str(midi_path),
        ]
        print(f"[fluidsynth] render fallback command: {' '.join(command)}", flush=True)
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"FluidSynth realtime and render fallback both failed: {error}")
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


class MacWebApp:
    def __init__(
        self,
        *,
        camera_index: int,
        midi_dir: str | Path,
        midi_manifest: str | Path | None,
        drum_dir: str | Path,
        drum_manifest: str | Path | None,
        arrangement_dir: str | Path,
        target_seconds: int,
        tempo_bpm: int,
        chord_program: int | None,
        player: MidiPlayer,
        soundfont_dir: str | Path | None = None,
        soundfont_manifest: str | Path | None = None,
        save_snapshot_dir: str | Path | None = None,
    ):
        self.cv2 = import_cv2()
        self.capture = self.cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open Mac webcam at index {camera_index}")

        self.midi_dir = Path(midi_dir).expanduser()
        self.midi_manifest = Path(midi_manifest).expanduser() if midi_manifest else None
        self.drum_dir = Path(drum_dir).expanduser()
        self.drum_manifest = Path(drum_manifest).expanduser() if drum_manifest else None
        self.arrangement_dir = Path(arrangement_dir).expanduser()
        self.target_seconds = target_seconds
        self.tempo_bpm = tempo_bpm
        self.chord_program = chord_program
        self.player = player
        self.soundfont_dir = Path(soundfont_dir).expanduser() if soundfont_dir else None
        self.soundfont_manifest = (
            Path(soundfont_manifest).expanduser() if soundfont_manifest else None
        )
        self.save_snapshot_dir = Path(save_snapshot_dir).expanduser() if save_snapshot_dir else None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.last_midi_path: Path | None = None
        self.running = True
        self.status = {
            "state": "idle",
            "message": "Press Snapshot and choose MIDI.",
            "filename": "",
            "path": "",
            "mood": "",
            "reason": "",
            "description": "",
            "midi_dir": str(self.midi_dir),
            "drum_dir": str(self.drum_dir),
            "fluidsynth": shutil.which("fluidsynth") or "",
            "soundfont": str(self.player.soundfont or ""),
            "soundfont_log": self.player.soundfont_log,
            "arrangement": {},
            "updated_at": "",
        }

    def start_camera(self) -> None:
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def shutdown(self) -> None:
        self.running = False
        self.player.stop()
        self.capture.release()

    def get_status(self) -> dict[str, str]:
        with self.status_lock:
            return dict(self.status)

    def snapshot_and_select(self) -> dict[str, str]:
        with self.status_lock:
            if self.status["state"] == "running":
                return dict(self.status)

        with self.frame_lock:
            if self.latest_frame is None:
                self._set_status(state="error", message="No webcam frame is available yet.")
                return self.get_status()
            snapshot = self.latest_frame.copy()

        self.worker = threading.Thread(target=self._select_and_play, args=(snapshot,), daemon=True)
        self.worker.start()
        self._set_status(state="running", message="Snapshot queued. Describing image with OpenAI...")
        return self.get_status()

    def replay_last(self) -> dict[str, str]:
        if not self.last_midi_path:
            self._set_status(state="idle", message="No MIDI has been selected yet.")
            return self.get_status()
        try:
            message = self.player.play(self.last_midi_path)
            self._set_status(state="playing", message=message)
        except Exception as exc:
            self._set_status(state="error", message=f"Replay failed: {exc}")
        return self.get_status()

    def snapshot_and_arrange(self) -> dict[str, str]:
        with self.status_lock:
            if self.status["state"] == "running":
                return dict(self.status)

        with self.frame_lock:
            if self.latest_frame is None:
                self._set_status(state="error", message="No webcam frame is available yet.")
                return self.get_status()
            snapshot = self.latest_frame.copy()

        self.worker = threading.Thread(target=self._select_arrange_and_play, args=(snapshot,), daemon=True)
        self.worker.start()
        self._set_status(state="running", message="Snapshot queued. Building chord + drum arrangement...")
        return self.get_status()

    def stop_midi(self) -> dict[str, str]:
        self.player.stop()
        self._set_status(state="idle", message="MIDI playback stopped.")
        return self.get_status()

    def mjpeg_frames(self):
        while self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(FRAME_INTERVAL)
                continue
            display = self._render_overlay(frame)
            ok, jpeg = self.cv2.imencode(".jpg", display)
            if not ok:
                continue
            yield jpeg.tobytes()
            time.sleep(FRAME_INTERVAL)

    def _camera_loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                self._set_status(state="error", message="Could not read webcam frame.")
                time.sleep(0.2)

    def _select_and_play(self, snapshot) -> None:
        try:
            from module import choose_midi_for_image
            from module.image_describer import extract_features

            features, _, _ = extract_features(snapshot)
            if self.save_snapshot_dir:
                self.save_snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = self.save_snapshot_dir / f"snapshot_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                self.cv2.imwrite(str(path), snapshot)

            self._set_status(state="running", message="Choosing MIDI from manifest...")
            selection = choose_midi_for_image(
                snapshot,
                self.midi_dir,
                features=features,
                manifest_path=self.midi_manifest,
            )
            self.last_midi_path = selection.path
            message = self.player.play(selection.path)
            self._set_status(
                state="playing",
                message=message,
                filename=selection.filename,
                path=str(selection.path),
                mood=selection.mood,
                reason=selection.reason,
                description=selection.image_description,
            )
        except Exception as exc:
            self._set_status(state="error", message=f"Selection failed: {exc}")

    def _select_arrange_and_play(self, snapshot) -> None:
        try:
            from concurrent.futures import ThreadPoolExecutor

            from module import (
                choose_midi_from_description,
                create_full_arrangement,
                default_arrangement_path,
                describe_image,
                list_midi_files,
            )
            from module.image_describer import extract_features
            from module.soundfont_picker import (
                load_soundfont_manifest,
                pick_soundfont,
            )

            features, _, _ = extract_features(snapshot)

            self._set_status(state="running", message="Describing image with OpenAI...")
            description = describe_image(snapshot, include_parts=True, max_parts=4)

            chord_candidates = list_midi_files(self.midi_dir, manifest_path=self.midi_manifest)
            drum_candidates = list_midi_files(self.drum_dir, manifest_path=self.drum_manifest)
            if not chord_candidates:
                raise FileNotFoundError(f"No chord MIDI files found in {self.midi_dir}")
            if not drum_candidates:
                raise FileNotFoundError(f"No drum MIDI files found in {self.drum_dir}")

            self._set_status(state="running", message="Choosing chord + drum MIDI in parallel...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                chord_future = pool.submit(
                    choose_midi_from_description,
                    description.description,
                    chord_candidates,
                    features=features,
                )
                drum_future = pool.submit(
                    choose_midi_from_description,
                    description.description,
                    drum_candidates,
                    features=features,
                )
                chord_selection = chord_future.result()
                drum_selection = drum_future.result()

            self._set_status(state="running", message="Building vibe-driven arrangement...")
            result = create_full_arrangement(
                chord_selection.path,
                drum_selection.path,
                default_arrangement_path(self.arrangement_dir),
                features=features,
                tempo_bpm=self.tempo_bpm,
                target_seconds=self.target_seconds,
            )

            soundfont_selection = None
            if self.soundfont_dir is not None:
                entries = load_soundfont_manifest(
                    self.soundfont_dir,
                    manifest_path=self.soundfont_manifest,
                )
                soundfont_selection = pick_soundfont(
                    result.vibe,
                    entries,
                    fallback=self.player.soundfont,
                )

            self.last_midi_path = result.output_path
            soundfont_path = (
                soundfont_selection.entry.path if soundfont_selection else None
            )
            message = self.player.play(result.output_path, soundfont=soundfont_path)
            arrangement = {
                "chords": chord_selection.filename,
                "drums": drum_selection.filename,
                **result.summary(),
            }
            if soundfont_selection is not None:
                arrangement["soundfont"] = {
                    "name": soundfont_selection.entry.name,
                    "path": str(soundfont_selection.entry.path),
                    "bucket": soundfont_selection.bucket,
                    "reason": soundfont_selection.reason,
                }
            self._set_status(
                state="playing",
                message=message,
                filename=result.output_path.name,
                path=str(result.output_path),
                mood=f"{chord_selection.mood} + {drum_selection.mood}",
                reason=f"Chords: {chord_selection.reason} Drums: {drum_selection.reason}",
                description=description.description,
                arrangement=arrangement,
            )
        except Exception as exc:
            self._set_status(state="error", message=f"Arrangement failed: {exc}")

    def _render_overlay(self, frame):
        display = frame.copy()
        lines = [
            "Browser controls: snapshot/select, replay, stop",
            f"MIDI dir: {self.midi_dir}",
            self.get_status().get("message", ""),
        ]
        y = 28
        for line in lines:
            self.cv2.putText(display, line, (12, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, self.cv2.LINE_AA)
            self.cv2.putText(display, line, (12, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 255, 140), 1, self.cv2.LINE_AA)
            y += 24
        return display

    def _set_status(self, **fields) -> None:
        with self.status_lock:
            self.status.update(fields)
            self.status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def make_handler(app: MacWebApp):
    class MacHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._serve_html()
            if path == "/video":
                return self._serve_video()
            if path == "/status":
                return self._send_json(app.get_status())
            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/snapshot":
                return self._send_json(app.snapshot_and_select(), status=202)
            if path == "/arrange":
                return self._send_json(app.snapshot_and_arrange(), status=202)
            if path == "/replay":
                return self._send_json(app.replay_last())
            if path == "/stop":
                return self._send_json(app.stop_midi())
            self.send_error(404)

        def _serve_html(self):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_video(self):
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for jpeg in app.mjpeg_frames():
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, data, status=200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return MacHandler


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is not installed. Run: pip install -r requirements.txt") from exc
    return cv2


def detect_soundfont() -> tuple[Path | None, list[str]]:
    log: list[str] = []
    env_path = os.getenv("SOUNDFONT_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        exists = path.exists()
        log.append(f"SOUNDFONT_PATH={path} exists={exists}")
        if exists:
            return path, log
    else:
        log.append("SOUNDFONT_PATH not set")

    for candidate in COMMON_SOUNDFONT_PATHS:
        path = Path(candidate).expanduser()
        log.append(f"checking common path {path} exists={path.exists()}")
        if path.exists():
            return path, log

    brew = shutil.which("brew")
    if brew:
        log.append(f"brew found at {brew}")
        for package in ("fluid-soundfont", "fluid-synth"):
            try:
                result = subprocess.run(
                    [brew, "--prefix", package],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                )
            except (subprocess.SubprocessError, OSError):
                log.append(f"brew --prefix {package} failed to run")
                continue

            prefix = result.stdout.strip()
            log.append(f"brew --prefix {package} -> {prefix or '(empty)'}")
            if not prefix:
                continue
            for name in ("FluidR3_GM.sf2", "FluidR3_GS.sf2"):
                for subdir in ("share/fluid-soundfont", "share/soundfonts", ""):
                    path = Path(prefix) / subdir / name
                    log.append(f"checking brew path {path} exists={path.exists()}")
                    if path.exists():
                        return path, log
    else:
        log.append("brew not found")

    local_candidates = [
        Path("soundfonts/GeneralUser_GS.sf2"),
        Path("soundfonts/FluidR3_GM.sf2"),
        Path("GeneralUser_GS.sf2"),
    ]
    for path in local_candidates:
        expanded = path.expanduser()
        log.append(f"checking local path {expanded} exists={expanded.exists()}")
        if expanded.exists():
            return expanded, log

    log.append("no SoundFont found")
    return None, log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser UI for camera-to-MIDI arrangement playback")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port")
    parser.add_argument("--midi-dir", default=str(DEFAULT_MIDI_DIR), help="Folder containing MIDI files")
    parser.add_argument("--midi-manifest", help="Optional midi_manifest.json path")
    parser.add_argument("--drum-dir", default=str(DEFAULT_DRUM_DIR), help="Folder containing drum MIDI files")
    parser.add_argument("--drum-manifest", help="Optional drum midi_manifest.json path")
    parser.add_argument("--arrangement-dir", default=str(DEFAULT_ARRANGEMENT_DIR), help="Where generated arrangements are saved")
    parser.add_argument("--target-seconds", type=int, default=60, help="Arrangement target duration")
    parser.add_argument("--tempo-bpm", type=int, default=110, help="Arrangement tempo")
    parser.add_argument("--chord-program", type=int, default=0, help="General MIDI program for chord layer; use -1 to preserve")
    parser.add_argument("--soundfont", default=os.getenv("SOUNDFONT_PATH"), help="Default SoundFont when no vibe-matched bank is available")
    parser.add_argument("--soundfont-dir", default="soundfonts", help="Folder of .sf2 banks for vibe-driven selection")
    parser.add_argument("--soundfont-manifest", help="Optional soundfont_manifest.json path; defaults to <soundfont-dir>/soundfont_manifest.json")
    parser.add_argument(
        "--rendered-audio-dir",
        default="generated_arrangements/rendered_audio",
        help="Where WAV fallback renders are saved if live FluidSynth audio fails",
    )
    parser.add_argument(
        "--fluidsynth-audio-driver",
        default=DEFAULT_FLUIDSYNTH_AUDIO_DRIVER,
        help="FluidSynth audio driver, e.g. coreaudio on macOS or alsa on Raspberry Pi",
    )
    parser.add_argument("--no-fluidsynth", action="store_true", help="Do not use FluidSynth for MIDI playback")
    parser.add_argument("--save-snapshots", help="Optional folder for captured webcam snapshots")
    parser.add_argument("--env-file", default=".env", help="Optional env file containing OPENAI_API_KEY")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv(args.env_file)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to your shell or .env before clicking Snapshot.")

    app = MacWebApp(
        camera_index=args.camera_index,
        midi_dir=args.midi_dir,
        midi_manifest=args.midi_manifest,
        drum_dir=args.drum_dir,
        drum_manifest=args.drum_manifest,
        arrangement_dir=args.arrangement_dir,
        target_seconds=args.target_seconds,
        tempo_bpm=args.tempo_bpm,
        chord_program=None if args.chord_program < 0 else args.chord_program,
        player=MidiPlayer(
            soundfont=args.soundfont,
            prefer_fluidsynth=not args.no_fluidsynth,
            audio_driver=args.fluidsynth_audio_driver,
            rendered_audio_dir=args.rendered_audio_dir,
        ),
        soundfont_dir=args.soundfont_dir,
        soundfont_manifest=args.soundfont_manifest,
        save_snapshot_dir=args.save_snapshots,
    )
    app.start_camera()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}/"
    print(f"Open {url} in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        app.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
