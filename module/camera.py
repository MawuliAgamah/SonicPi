"""Camera capture helpers shared by project modes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np


CameraBackend = Literal["auto", "opencv", "picamera2"]


@dataclass(frozen=True)
class CapturedFrame:
    """Single BGR OpenCV frame captured from a camera backend."""

    frame: np.ndarray
    backend: str


def capture_frame(
    *,
    camera_index: int = 0,
    backend: CameraBackend = "auto",
    warmup_seconds: float = 1.0,
) -> CapturedFrame:
    """Capture one BGR frame from PiCamera2 or OpenCV.

    ``auto`` prefers PiCamera2 when available, then falls back to OpenCV. That
    keeps Raspberry Pi deployment simple without breaking laptop development.
    """

    if backend not in {"auto", "opencv", "picamera2"}:
        raise ValueError(f"Unsupported camera backend: {backend}")

    if backend in {"auto", "picamera2"}:
        try:
            return _capture_picamera2(warmup_seconds=warmup_seconds)
        except Exception:
            if backend == "picamera2":
                raise

    return _capture_opencv(camera_index=camera_index, warmup_seconds=warmup_seconds)


def _capture_picamera2(*, warmup_seconds: float) -> CapturedFrame:
    try:
        import cv2
        from picamera2 import Picamera2
    except ModuleNotFoundError as exc:
        raise RuntimeError("PiCamera2 capture is not available") from exc

    camera = Picamera2()
    try:
        camera.start()
        if warmup_seconds > 0:
            time.sleep(warmup_seconds)
        raw = camera.capture_array()
    finally:
        camera.stop()

    if raw is None or raw.size == 0:
        raise RuntimeError("PiCamera2 returned an empty frame")

    if len(raw.shape) == 3 and raw.shape[2] >= 3:
        frame = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    return CapturedFrame(frame=frame, backend="picamera2")


def _capture_opencv(*, camera_index: int, warmup_seconds: float) -> CapturedFrame:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is not installed. Run: pip install -r requirements.txt") from exc

    capture = cv2.VideoCapture(camera_index)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera at index {camera_index}")
        deadline = time.monotonic() + max(0.0, warmup_seconds)
        ok = False
        frame = None
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            time.sleep(0.05)
        if not ok or frame is None:
            ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"Could not read OpenCV camera frame at index {camera_index}")
        return CapturedFrame(frame=frame, backend="opencv")
    finally:
        capture.release()
