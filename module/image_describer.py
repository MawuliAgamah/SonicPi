"""OpenCV image extraction helpers plus OpenAI vision description.

The extraction mirrors the useful visual signals already used by ``synth.py``
and adds crop extraction so a model can inspect both the whole image and
important sub-regions.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_FRAME_SIZE = (320, 240)
DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class ImagePart:
    """A named image crop produced by OpenCV."""

    name: str
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    kind: str
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "name": self.name,
            "kind": self.kind,
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ImageDescription:
    """OpenAI description result with the local OpenCV context used."""

    description: str
    features: dict[str, float | int]
    parts: list[dict[str, Any]]
    model: str


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR OpenCV array."""

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channel image, got shape {image.shape}")
    return image


def extract_features(
    image: np.ndarray,
    previous_gray: np.ndarray | None = None,
    frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
) -> tuple[dict[str, float | int], dict[str, Any], np.ndarray]:
    """Extract the same core visual features used by ``synth.py``.

    ``previous_gray`` is optional. Pass the grayscale frame returned from the
    prior call when processing video frames so motion can be calculated.
    """

    frame = cv2.resize(_ensure_bgr(image), frame_size)
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
    direction = float(sobel_x / total_energy)

    motion = 0.0
    if previous_gray is not None:
        previous = cv2.resize(previous_gray, frame_size)
        diff = cv2.absdiff(previous, gray)
        motion = float(np.mean(diff))

    features: dict[str, float | int] = {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "dominant_hue": dominant_hue,
        "edge_density": edge_density,
        "contour_count": contour_count,
        "direction": direction,
        "motion": motion,
    }
    aux = {
        "edges": edges,
        "contours": contours,
        "frame_small": frame,
    }
    return features, aux, gray


def extract_image_parts(
    image: np.ndarray,
    *,
    max_contour_parts: int = 6,
    min_area_ratio: float = 0.01,
    include_quadrants: bool = True,
    include_center: bool = True,
) -> list[ImagePart]:
    """Extract whole-image, fixed-region, and contour-based crops.

    The first returned part is always the full image. Contour crops are sorted
    from largest to smallest and de-duplicated by bounding-box overlap.
    """

    image = _ensure_bgr(image)
    height, width = image.shape[:2]
    parts = [
        ImagePart(
            name="full_image",
            image=image.copy(),
            bbox=(0, 0, width, height),
            kind="full",
            metadata={"source": "original"},
        )
    ]

    if include_center:
        crop_w = max(1, int(width * 0.5))
        crop_h = max(1, int(height * 0.5))
        x = (width - crop_w) // 2
        y = (height - crop_h) // 2
        parts.append(_make_part("center_crop", image, (x, y, crop_w, crop_h), "fixed"))

    if include_quadrants:
        half_w = width // 2
        half_h = height // 2
        quadrants = [
            ("top_left", (0, 0, half_w, half_h)),
            ("top_right", (half_w, 0, width - half_w, half_h)),
            ("bottom_left", (0, half_h, half_w, height - half_h)),
            ("bottom_right", (half_w, half_h, width - half_w, height - half_h)),
        ]
        for name, bbox in quadrants:
            parts.append(_make_part(name, image, bbox, "fixed"))

    parts.extend(
        _extract_contour_parts(
            image,
            max_parts=max_contour_parts,
            min_area_ratio=min_area_ratio,
        )
    )
    return parts


def _make_part(name: str, image: np.ndarray, bbox: tuple[int, int, int, int], kind: str) -> ImagePart:
    x, y, w, h = _clip_bbox(bbox, image.shape[1], image.shape[0])
    crop = image[y : y + h, x : x + w].copy()
    return ImagePart(
        name=name,
        image=crop,
        bbox=(x, y, w, h),
        kind=kind,
        metadata={"area_ratio": round((w * h) / (image.shape[0] * image.shape[1]), 4)},
    )


def _extract_contour_parts(
    image: np.ndarray,
    *,
    max_parts: int,
    min_area_ratio: float,
) -> list[ImagePart]:
    height, width = image.shape[:2]
    image_area = width * height
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area / image_area < min_area_ratio:
            continue
        candidates.append((area, _expand_bbox((x, y, w, h), width, height, padding_ratio=0.12)))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, int, int, int]] = []
    for _, bbox in candidates:
        if any(_bbox_iou(bbox, existing) > 0.45 for existing in selected):
            continue
        selected.append(bbox)
        if len(selected) >= max_parts:
            break

    contour_parts = []
    for index, bbox in enumerate(selected, start=1):
        part = _make_part(f"contour_{index}", image, bbox, "contour")
        part.metadata["rank"] = index
        contour_parts.append(part)
    return contour_parts


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    return _clip_bbox((x - pad_x, y - pad_y, w + pad_x * 2, h + pad_y * 2), width, height)


def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def save_parts(parts: list[ImagePart], output_dir: str | Path) -> list[Path]:
    """Write extracted parts to ``output_dir`` and return their paths."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for part in parts:
        path = output_path / f"{part.name}.jpg"
        ok = cv2.imwrite(str(path), part.image)
        if not ok:
            raise OSError(f"Could not write image part: {path}")
        saved_paths.append(path)
    return saved_paths


def describe_image(
    image_or_path: str | Path | np.ndarray,
    *,
    api_key: str | None = None,
    model: str | None = None,
    include_parts: bool = True,
    max_parts: int = 8,
    prompt: str | None = None,
    timeout: float = 60.0,
) -> ImageDescription:
    """Extract image context and ask OpenAI for a plain-language description."""

    image = load_image(image_or_path) if isinstance(image_or_path, (str, Path)) else _ensure_bgr(image_or_path)
    features, _, _ = extract_features(image)
    all_parts = extract_image_parts(image) if include_parts else [
        ImagePart(
            name="full_image",
            image=image.copy(),
            bbox=(0, 0, image.shape[1], image.shape[0]),
            kind="full",
            metadata={"source": "original"},
        )
    ]
    selected_parts = _select_parts_for_model(all_parts, max_parts=max_parts)
    summaries = [part.summary() for part in selected_parts]

    request_prompt = prompt or _default_prompt(features, summaries)
    response = _call_openai_vision(
        parts=selected_parts,
        prompt=request_prompt,
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        timeout=timeout,
    )
    return ImageDescription(
        description=response,
        features=features,
        parts=summaries,
        model=model or DEFAULT_MODEL,
    )


def _select_parts_for_model(parts: list[ImagePart], *, max_parts: int) -> list[ImagePart]:
    if max_parts < 1:
        raise ValueError("max_parts must be at least 1")

    full = [part for part in parts if part.kind == "full"]
    fixed = [part for part in parts if part.kind == "fixed"]
    contours = [part for part in parts if part.kind == "contour"]
    ordered = full + fixed[:2] + contours
    return ordered[:max_parts]


def _default_prompt(features: dict[str, float | int], parts: list[dict[str, Any]]) -> str:
    feature_summary = {
        key: round(value, 3) if isinstance(value, float) else value
        for key, value in features.items()
    }
    return (
        "Describe the image clearly and concisely. Include the main subject, "
        "setting, notable objects, colors, composition, visible text, and any "
        "important uncertainty. You will receive the full image plus optional "
        "OpenCV-extracted crops. Use the crops only to improve the overall "
        "description; do not describe them as separate files unless that is useful.\n\n"
        f"OpenCV feature summary:\n{json.dumps(feature_summary, indent=2)}\n\n"
        f"Image parts sent:\n{json.dumps(parts, indent=2)}"
    )


def _call_openai_vision(
    *,
    parts: list[ImagePart],
    prompt: str,
    api_key: str | None,
    model: str,
    timeout: float,
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required to describe images with OpenAI")

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for part in parts:
        content.append(
            {
                "type": "input_image",
                "image_url": _image_to_data_url(part.image),
            }
        )

    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 700,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    text = _extract_response_text(response_data)
    if not text:
        raise RuntimeError(f"OpenAI API response did not include output text: {response_data}")
    return text


def _image_to_data_url(image: np.ndarray, *, jpeg_quality: int = 90) -> str:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, encoded = cv2.imencode(".jpg", image, params)
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _extract_response_text(response_data: dict[str, Any]) -> str:
    output_text = response_data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for output in response_data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe an image with OpenCV crops and OpenAI vision")
    parser.add_argument("image", help="Path to the image to describe")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--no-parts", action="store_true", help="Send only the full image")
    parser.add_argument("--max-parts", type=int, default=8, help="Maximum image parts to send")
    parser.add_argument("--save-parts", help="Directory where extracted crops should be saved")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    image = load_image(args.image)

    if args.save_parts:
        saved = save_parts(extract_image_parts(image), args.save_parts)
        print(f"Saved {len(saved)} image parts to {args.save_parts}")

    result = describe_image(
        image,
        model=args.model,
        include_parts=not args.no_parts,
        max_parts=args.max_parts,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "description": result.description,
                    "features": result.features,
                    "parts": result.parts,
                    "model": result.model,
                },
                indent=2,
            )
        )
    else:
        print(result.description)


if __name__ == "__main__":
    main()
