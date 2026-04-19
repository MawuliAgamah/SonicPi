"""Browser-controlled ambient synth driven by webcam features."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from module.ambient_model import (
    VibeProfile,
    ambient_state_for,
    pick_modal_key,
    smooth_vibe,
    vibe_from_feature_dict,
)
from module.image_describer import extract_features
from module.pyo_ambient_engine import PyoAmbientEngine
from module.sounddevice_ambient_engine import SoundDeviceAmbientEngine


DEFAULT_PORT = 8091
FRAME_INTERVAL = 0.05


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Ambient Camera Synth</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #101211;
  color: #edf2ed;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
}
.stage {
  min-width: 0;
  padding: 20px;
}
img {
  display: block;
  width: 100%;
  height: calc(100vh - 40px);
  object-fit: contain;
  background: #030403;
  border: 1px solid #2c332f;
}
aside {
  border-left: 1px solid #2c332f;
  padding: 20px;
  background: #171b19;
}
h1 {
  margin: 0 0 18px;
  font-size: 20px;
  font-weight: 650;
  letter-spacing: 0;
}
.controls {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 18px;
}
button {
  min-height: 42px;
  border: 1px solid #3b4740;
  border-radius: 6px;
  background: #24302a;
  color: #f3f8f2;
  font: inherit;
  font-weight: 650;
  cursor: pointer;
}
button:hover { background: #2d3c34; }
button:disabled { cursor: wait; opacity: .6; }
.wide { grid-column: span 2; }
.status {
  display: grid;
  gap: 14px;
}
.row {
  display: grid;
  grid-template-columns: 92px minmax(0, 1fr);
  gap: 10px;
  align-items: center;
  font-size: 14px;
}
.label { color: #aab7ad; }
.value {
  min-width: 0;
  overflow-wrap: anywhere;
}
.bar {
  height: 9px;
  background: #303833;
  border-radius: 999px;
  overflow: hidden;
}
.fill {
  width: 0%;
  height: 100%;
  background: #8bd4a2;
}
.pillbox {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.pill {
  border: 1px solid #3d4a43;
  border-radius: 999px;
  padding: 4px 8px;
  color: #dfe9e1;
  background: #202821;
  font-size: 12px;
}
.message {
  margin-top: 16px;
  color: #b8c8ba;
  font-size: 13px;
  line-height: 1.4;
}
@media (max-width: 820px) {
  main { grid-template-columns: 1fr; }
  img { height: auto; max-height: 58vh; }
  aside { border-left: 0; border-top: 1px solid #2c332f; }
}
</style>
</head>
<body>
<main>
  <section class="stage"><img src="/video" alt="webcam stream"></section>
  <aside>
    <h1>Ambient Camera Synth</h1>
    <div class="controls">
      <button id="start">Start</button>
      <button id="hold">Hold</button>
      <button id="reset">Reset key</button>
      <button id="stop">Stop</button>
    </div>
    <section class="status">
      <div class="row"><div class="label">audio</div><div class="value" id="audio">stopped</div></div>
      <div class="row"><div class="label">key</div><div class="value" id="key">-</div></div>
      <div class="row"><div class="label">chord</div><div class="value" id="chord">-</div></div>
      <div class="row"><div class="label">next</div><div class="value" id="next">-</div></div>
      <div class="row"><div class="label">brightness</div><div class="bar"><div class="fill" id="brightness"></div></div></div>
      <div class="row"><div class="label">color</div><div class="bar"><div class="fill" id="color"></div></div></div>
      <div class="row"><div class="label">energy</div><div class="bar"><div class="fill" id="energy"></div></div></div>
      <div class="row"><div class="label">complexity</div><div class="bar"><div class="fill" id="complexity"></div></div></div>
      <div class="row"><div class="label">layers</div><div class="pillbox" id="layers"></div></div>
    </section>
    <div class="message" id="message"></div>
  </aside>
</main>
<script>
async function post(path) {
  const response = await fetch(path, { method: 'POST' });
  await response.json();
  refresh();
}

function setBar(id, value) {
  document.getElementById(id).style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`;
}

function setLayers(layers) {
  const root = document.getElementById('layers');
  root.innerHTML = '';
  for (const layer of layers || []) {
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.innerText = layer;
    root.appendChild(pill);
  }
}

async function refresh() {
  try {
    const response = await fetch('/status');
    const data = await response.json();
    document.getElementById('audio').innerText = data.audio_state + (data.hold ? ' / held' : '');
    document.getElementById('key').innerText = data.key || '-';
    document.getElementById('chord').innerText = data.chord || '-';
    document.getElementById('next').innerText = `${data.next_chord_in || 0}s`;
    document.getElementById('message').innerText = data.message || '';
    setBar('brightness', data.vibe?.brightness);
    setBar('color', data.vibe?.color);
    setBar('energy', data.vibe?.energy);
    setBar('complexity', data.vibe?.complexity);
    setLayers(data.layers);
  } catch (error) {}
}

document.getElementById('start').addEventListener('click', () => post('/start'));
document.getElementById('hold').addEventListener('click', () => post('/hold'));
document.getElementById('reset').addEventListener('click', () => post('/reset-key'));
document.getElementById('stop').addEventListener('click', () => post('/stop'));
setInterval(refresh, 700);
refresh();
</script>
</body>
</html>"""


class AmbientWebApp:
    def __init__(
        self,
        *,
        camera_index: int,
        smoothing_seconds: float,
        enable_arpeggio: bool,
        enable_texture: bool,
        master_gain: float,
        engine_name: str,
        sample_rate: int,
        blocksize: int,
        audio_device: str | int | None,
    ):
        self.cv2 = import_cv2()
        self.capture = self.cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")

        self.smoothing_seconds = max(0.1, smoothing_seconds)
        self.enable_arpeggio = enable_arpeggio
        self.enable_texture = enable_texture
        self.master_gain = master_gain
        self.engine_name = engine_name
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.audio_device = audio_device

        self.frame_lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.latest_frame = None
        self.latest_features: dict[str, float | int] = {}
        self.smoothed_vibe: VibeProfile | None = None
        self.held_vibe: VibeProfile | None = None
        self.held_motion = 0.0
        self.hold = False
        self.key = pick_modal_key(VibeProfile(0.25, 0.5, 0.35, 0.25))
        self.engine: Any | None = None
        self.running = True
        self.started_at = time.monotonic()
        self.message = "Stopped."

    def start_camera(self) -> None:
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def shutdown(self) -> None:
        self.running = False
        if self.engine is not None:
            self.engine.stop()
        self.capture.release()

    def start_audio(self) -> dict[str, object]:
        with self.state_lock:
            vibe = self.active_vibe
            if self.engine is None:
                self.key = pick_modal_key(vibe)
                self.engine = self._build_engine(vibe)
            try:
                self.engine.start()
                self.engine.set_vibe(vibe, motion=self._motion_value())
                self.message = "Ambient engine running."
            except Exception as exc:
                if self.engine_name == "auto" and isinstance(self.engine, SoundDeviceAmbientEngine):
                    self.engine.stop()
                    self.engine = self._build_engine(vibe, prefer_pyo=True)
                    try:
                        self.engine.start()
                        self.engine.set_vibe(vibe, motion=self._motion_value())
                        self.message = "Ambient engine running with pyo fallback."
                    except Exception as fallback_exc:
                        self.message = f"sounddevice failed: {exc}; pyo failed: {fallback_exc}"
                else:
                    self.message = str(exc)
            return self.status()

    def stop_audio(self) -> dict[str, object]:
        with self.state_lock:
            if self.engine is not None:
                self.engine.stop()
            self.message = "Stopped."
            return self.status()

    def toggle_hold(self) -> dict[str, object]:
        with self.state_lock:
            self.hold = not self.hold
            self.held_vibe = self.active_vibe if self.hold else None
            self.held_motion = self._motion_value() if self.hold else 0.0
            self.message = "Holding current vibe." if self.hold else "Following camera vibe."
            if self.engine is not None:
                self.engine.set_vibe(self.active_vibe, motion=self._motion_value())
            return self.status()

    def reset_key(self) -> dict[str, object]:
        with self.state_lock:
            vibe = self.active_vibe
            self.key = pick_modal_key(vibe)
            if self.engine is not None:
                self.engine.reset_key(vibe)
            self.message = f"Key reset to {self.key.label}."
            return self.status()

    @property
    def active_vibe(self) -> VibeProfile:
        fallback = VibeProfile(0.25, 0.5, 0.35, 0.25)
        if self.hold and self.held_vibe is not None:
            return self.held_vibe
        return self.smoothed_vibe or fallback

    def status(self) -> dict[str, object]:
        with self.state_lock:
            vibe = self.active_vibe
            elapsed = (
                self.engine.elapsed_seconds
                if self.engine is not None and self.engine.running
                else time.monotonic() - self.started_at
            )
            music_state = (
                self.engine.music_state()
                if self.engine is not None and self.engine.running
                else ambient_state_for(
                    vibe,
                    self.key,
                    elapsed_seconds=elapsed,
                    enable_arpeggio=self.enable_arpeggio,
                    enable_texture=self.enable_texture,
                )
            )
            interval = music_state.chord_change_seconds
            next_chord = interval - (elapsed % interval)
            engine_status = self.engine.status() if self.engine is not None else None
            return {
                "audio_state": "running" if engine_status and engine_status.running else "stopped",
                "hold": self.hold,
                "message": self.message if self.message else (engine_status.message if engine_status else ""),
                "vibe": {
                    "energy": round(vibe.energy, 3),
                    "brightness": round(vibe.brightness, 3),
                    "color": round(vibe.color, 3),
                    "complexity": round(vibe.complexity, 3),
                },
                "key": music_state.key.label,
                "chord": music_state.chord.name,
                "roman": music_state.chord.roman,
                "layers": list(music_state.layers),
                "tempo_bpm": music_state.tempo_bpm,
                "next_chord_in": round(next_chord, 1),
                "features": {
                    key: round(value, 3) if isinstance(value, float) else value
                    for key, value in self.latest_features.items()
                },
            }

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
        last_update = time.monotonic()
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                self.message = "Could not read webcam frame."
                time.sleep(0.2)
                continue

            features, _, _ = extract_features(frame)
            current = vibe_from_feature_dict(features)
            now = time.monotonic()
            dt = max(0.001, now - last_update)
            last_update = now
            alpha = min(1.0, dt / self.smoothing_seconds)

            with self.state_lock:
                self.latest_features = features
                self.smoothed_vibe = smooth_vibe(self.smoothed_vibe, current, alpha=alpha)
                if self.engine is not None and self.engine.running:
                    self.engine.set_vibe(self.active_vibe, motion=self._motion_value())

            with self.frame_lock:
                self.latest_frame = frame

            time.sleep(FRAME_INTERVAL)

    def _motion_value(self) -> float:
        if self.hold:
            return self.held_motion
        return float(self.latest_features.get("motion", 0.0))

    def _build_engine(self, vibe: VibeProfile, *, prefer_pyo: bool = False):
        if self.engine_name == "pyo" or prefer_pyo:
            return PyoAmbientEngine(
                initial_vibe=vibe,
                key=self.key,
                enable_arpeggio=self.enable_arpeggio,
                enable_texture=self.enable_texture,
                master_gain=self.master_gain,
            )
        return SoundDeviceAmbientEngine(
            initial_vibe=vibe,
            key=self.key,
            enable_arpeggio=self.enable_arpeggio,
            enable_texture=self.enable_texture,
            master_gain=self.master_gain,
            sample_rate=self.sample_rate,
            blocksize=self.blocksize,
            audio_device=self.audio_device,
        )

    def _render_overlay(self, frame):
        display = frame.copy()
        status = self.status()
        lines = [
            f"{status['audio_state']} | {status['key']} | {status['chord']}",
            f"energy {status['vibe']['energy']:.2f} brightness {status['vibe']['brightness']:.2f} color {status['vibe']['color']:.2f}",
            "hold" if status["hold"] else "follow",
        ]
        y = 28
        for line in lines:
            self.cv2.putText(display, line, (12, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, self.cv2.LINE_AA)
            self.cv2.putText(display, line, (12, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 190), 1, self.cv2.LINE_AA)
            y += 24
        return display


def make_handler(app: AmbientWebApp):
    class AmbientHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._serve_html()
            if path == "/video":
                return self._serve_video()
            if path == "/status":
                return self._send_json(app.status())
            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/start":
                return self._send_json(app.start_audio(), status=202)
            if path == "/stop":
                return self._send_json(app.stop_audio())
            if path == "/hold":
                return self._send_json(app.toggle_hold())
            if path == "/reset-key":
                return self._send_json(app.reset_key())
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

    return AmbientHandler


def import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is not installed. Run: pip install -r requirements.txt") from exc
    return cv2


def _parse_audio_device(value: str | None) -> str | int | None:
    if value is None or not value.strip():
        return None
    stripped = value.strip()
    try:
        return int(stripped)
    except ValueError:
        return stripped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera-driven ambient synth")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port")
    parser.add_argument(
        "--engine",
        choices=("sounddevice", "pyo", "auto"),
        default="sounddevice",
        help="Audio engine to use",
    )
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate for sounddevice")
    parser.add_argument("--blocksize", type=int, default=512, help="Audio callback block size for sounddevice")
    parser.add_argument("--audio-device", help="Optional sounddevice output device name or index")
    parser.add_argument("--smoothing-seconds", type=float, default=5.0, help="Vibe smoothing window")
    parser.add_argument("--no-arpeggio", action="store_true", help="Disable sparse arpeggio layer")
    parser.add_argument("--no-texture", action="store_true", help="Disable noise/shimmer texture layer")
    parser.add_argument("--master-gain", type=float, default=0.45, help="Audio master gain")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = AmbientWebApp(
        camera_index=args.camera_index,
        smoothing_seconds=args.smoothing_seconds,
        enable_arpeggio=not args.no_arpeggio,
        enable_texture=not args.no_texture,
        master_gain=args.master_gain,
        engine_name=args.engine,
        sample_rate=args.sample_rate,
        blocksize=args.blocksize,
        audio_device=_parse_audio_device(args.audio_device),
    )
    app.start_camera()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}/"
    print(f"Open {url} in your browser")
    print(f"Audio engine: {args.engine}. Audio starts after pressing Start.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        app.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
