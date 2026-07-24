"""
detector.py – YOLOv8 object detection + ByteTrack tracking for VisionGuard AI.

Responsibilities
----------------
- Load the YOLOv8n model once (cached).
- Run model.track() with ByteTrack on a single frame.
- Return structured detection results.
- Annotate frames with bounding boxes, class labels, confidence, and track IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils import class_colour, track_id_colour, get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Device selection helper
# ---------------------------------------------------------------------------

def get_optimal_device() -> str:
    """Return 'cuda' if PyTorch CUDA is available, otherwise 'cpu'."""
    if torch.cuda.is_available():
        dev = "cuda"
        _log.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
    else:
        dev = "cpu"
        _log.info("CUDA not available. Using CPU for inference.")
    return dev


# ---------------------------------------------------------------------------
# Detection result type
# ---------------------------------------------------------------------------

class Detection:
    """Structured result for a single detected object in one frame."""

    __slots__ = (
        "frame_idx",
        "track_id",
        "class_name",
        "confidence",
        "bbox",          # (x1, y1, x2, y2) in pixel coords
    )

    def __init__(
        self,
        frame_idx: int,
        track_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        self.frame_idx = frame_idx
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "frame": self.frame_idx,
            "track_id": self.track_id,
            "object": self.class_name,
            "confidence": self.confidence,
            "bbox": f"({x1},{y1},{x2},{y2})",
        }


# ---------------------------------------------------------------------------
# Model loader (loaded once, cached for the Python process lifetime)
# ---------------------------------------------------------------------------

_MODEL_CACHE: Optional[YOLO] = None
_MODEL_WEIGHTS: Optional[str] = None
_MODEL_DEVICE: Optional[str] = None


def load_model(weights: str = "yolov8s.pt", device: Optional[str] = None) -> YOLO:
    """
    Load and cache the YOLOv8 model.

    Automatically selects CUDA GPU if available, or CPU otherwise.

    Parameters
    ----------
    weights:
        Path or name of the YOLOv8 weights file.
    device:
        Optional explicit device string ('cuda' or 'cpu').

    Returns
    -------
    Loaded YOLO model instance.
    """
    global _MODEL_CACHE, _MODEL_WEIGHTS, _MODEL_DEVICE
    target_device = device or get_optimal_device()

    if (
        _MODEL_CACHE is None
        or _MODEL_WEIGHTS != weights
        or _MODEL_DEVICE != target_device
    ):
        _log.info("Loading YOLOv8 model '%s' on device '%s'", weights, target_device)
        model = YOLO(weights)
        model.to(target_device)
        _MODEL_CACHE = model
        _MODEL_WEIGHTS = weights
        _MODEL_DEVICE = target_device
        _log.info(
            "Model loaded successfully on %s. Classes: %d",
            target_device,
            len(model.names),
        )
    return _MODEL_CACHE


def reset_model_cache() -> None:
    """Clear the cached model (useful for testing or reloading)."""
    global _MODEL_CACHE, _MODEL_WEIGHTS, _MODEL_DEVICE
    _MODEL_CACHE = None
    _MODEL_WEIGHTS = None
    _MODEL_DEVICE = None
    _log.info("Model cache cleared.")


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def detect_frame(
    model: YOLO,
    frame: np.ndarray,
    frame_idx: int,
    conf_threshold: float = 0.60,
    iou_threshold: float = 0.60,
    device: Optional[str] = None,
    min_box_area: int = 300,
    allowed_classes: Optional[Set[str]] = None,
) -> List[Detection]:
    """
    Run YOLOv8 tracking on a single *frame* and return structured detections.

    Parameters
    ----------
    model:
        Loaded YOLO model.
    frame:
        BGR image as a NumPy array.
    frame_idx:
        Index of the current frame (used for record keeping only).
    conf_threshold:
        Minimum confidence to keep a detection.
    iou_threshold:
        IoU threshold for non-max suppression.
    device:
        Inference device ('cuda' or 'cpu').
    min_box_area:
        Minimum pixel area (w * h) for a bounding box to be valid.

    Returns
    -------
    List of Detection objects for this frame.
    """
    target_device = device or _MODEL_DEVICE or get_optimal_device()
    try:
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            device=target_device,
            verbose=False,
        )
    except Exception as exc:
        _log.error("model.track() failed on frame %d: %s", frame_idx, exc)
        return []

    detections: List[Detection] = []

    if not results or results[0].boxes is None:
        return detections

    result = results[0]
    boxes = result.boxes

    # Guard: no detections
    if boxes.xyxy is None or len(boxes.xyxy) == 0:
        return detections

    # Extract per-box data
    xyxy_all = boxes.xyxy.cpu().numpy().astype(int)           # shape (N,4)
    conf_all = boxes.conf.cpu().numpy().astype(float)          # shape (N,)
    cls_all = boxes.cls.cpu().numpy().astype(int)              # shape (N,)

    # Track IDs – may be None if tracking hasn't assigned them yet
    if boxes.id is not None:
        id_all = boxes.id.cpu().numpy().astype(int)
    else:
        id_all = np.arange(len(xyxy_all), dtype=int)          # fallback

    class_names: Dict[int, str] = model.names                 # {class_id: name}

    for i in range(len(xyxy_all)):
        x1, y1, x2, y2 = xyxy_all[i]
        conf = float(conf_all[i])
        cls_id = int(cls_all[i])
        track_id = int(id_all[i]) if i < len(id_all) else i

        w = x2 - x1
        h = y2 - y1

        # Ignore detections with confidence below threshold, small dimensions, or tiny box area
        if conf < conf_threshold or w < 10 or h < 10 or (w * h) < min_box_area:
            continue

        class_name = class_names.get(cls_id, f"class_{cls_id}")

        # Filter by allowed_classes if specified (for Detection Modes e.g. People Only, Vehicles Only)
        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        detections.append(
            Detection(
                frame_idx=frame_idx,
                track_id=track_id,
                class_name=class_name,
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            )
        )

    return detections


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    detections: List[Detection],
    in_place: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on *frame* for every detection.

    Labels format:  «ClassName | ID {track_id} | {conf}%»

    Each unique track ID gets a distinct colour so the same object is always
    shown in the same colour regardless of which frame is being drawn.

    Parameters
    ----------
    frame:
        BGR frame to draw on.
    detections:
        List of Detection objects for this frame.
    in_place:
        If True, draw directly on *frame* without copying (caller must own
        the array exclusively).  Saves one full-frame allocation per frame.

    Returns
    -------
    Annotated BGR frame.  Same object as *frame* when in_place=True.
    """
    annotated = frame if in_place else frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        colour = track_id_colour(det.track_id)

        # Bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness=2)

        # Label text
        label = f"{det.class_name} | ID {det.track_id} | {det.confidence * 100:.0f}%"

        # Dynamic font scale based on box size
        box_h = max(y2 - y1, 1)
        font_scale = max(0.35, min(0.65, box_h / 120.0))
        thickness = 1 if font_scale < 0.55 else 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Background rectangle for label
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        label_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_w + 4, label_y2),
            colour,
            cv2.FILLED,
        )

        # Dark text on coloured background
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y2 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )

    return annotated
