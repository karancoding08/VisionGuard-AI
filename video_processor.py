"""
video_processor.py – End-to-end video reading, detection, and writing pipeline
for VisionGuard AI.

Responsibilities
----------------
- Open an input video file with OpenCV.
- Iterate every frame.
- Call the detector on each frame.
- Annotate each frame.
- Write annotated frames to an output video file.
- Accumulate statistics and detection records.
- Report per-frame progress via a callback.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import cv2
import numpy as np

from detector import Detection, annotate_frame, detect_frame, load_model
from utils import (
    ensure_dirs,
    export_detections_csv,
    format_seconds,
    generate_ai_summary,
    get_logger,
    safe_divide,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Processing result container
# ---------------------------------------------------------------------------


class ProcessingResult:
    """Holds every artefact produced after processing a video."""

    def __init__(self) -> None:
        self.frames_processed: int = 0
        self.processing_time: float = 0.0        # seconds
        self.fps_input: float = 0.0              # source video FPS
        self.resolution: str = "N/A"             # "WxH"
        self.total_detections: int = 0
        self.all_records: List[Dict] = []        # one dict per detection per frame
        self.unique_track_ids: Dict[str, Set[int]] = {}  # class_name → set of track IDs
        self.confidence_sum: float = 0.0
        self.output_video_path: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.ai_summary: str = ""
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Derived properties & Task 2 aliases                                 #
    # ------------------------------------------------------------------ #

    @property
    def total_frames(self) -> int:
        return self.frames_processed

    @property
    def fps(self) -> float:
        return self.fps_input

    @property
    def objects_detected(self) -> int:
        return self.total_detections

    @property
    def unique_objects(self) -> int:
        return self.unique_objects_count

    @property
    def unique_objects_count(self) -> int:
        """Total number of unique physical objects across all classes."""
        return sum(len(ids) for ids in self.unique_track_ids.values())

    @property
    def object_counts(self) -> Dict[str, int]:
        """Return {class_name: unique_count} for every detected class."""
        return {cls: len(ids) for cls, ids in self.unique_track_ids.items()}

    @property
    def average_confidence(self) -> float:
        return safe_divide(self.confidence_sum, self.total_detections)

    @property
    def most_frequent_object(self) -> str:
        counts = self.object_counts
        if not counts:
            return "N/A"
        return max(counts, key=lambda k: counts[k])

    @property
    def detections(self) -> List[Dict]:
        return self.all_records

    @property
    def output_path(self) -> Optional[Path]:
        return self.output_video_path

    @property
    def effective_fps(self) -> float:
        """Effective processing throughput (frames / elapsed time)."""
        return safe_divide(self.frames_processed, self.processing_time)

    def to_stats_dict(self) -> Dict:
        return {
            "total_frames": self.total_frames,
            "frames_processed": self.frames_processed,
            "processing_time": self.processing_time,
            "resolution": self.resolution,
            "fps": self.fps,
            "total_detections": self.total_detections,
            "objects_detected": self.objects_detected,
            "unique_objects": self.unique_objects,
            "unique_objects_count": self.unique_objects_count,
            "average_confidence": self.average_confidence,
            "object_counts": self.object_counts,
            "most_frequent_object": self.most_frequent_object,
            "detections": self.detections,
            "output_path": self.output_path,
        }


# ---------------------------------------------------------------------------
# Video validator
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov")


def validate_video(path: str | Path) -> Optional[str]:
    """
    Check that *path* points to a readable, non-empty video file.

    Returns
    -------
    None if valid, or an error message string if invalid.
    """
    path = Path(path)

    if not path.exists():
        return f"File not found: {path}"

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return f"OpenCV could not open file: {path.name}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return f"Video appears to be empty (0 frames): {path.name}"

        ret, frame = cap.read()
        if not ret or frame is None:
            return f"Could not read the first frame – file may be corrupted: {path.name}"
    finally:
        cap.release()

    return None  # valid


MODE_ALLOWED_CLASSES: Dict[str, Optional[Set[str]]] = {
    "All Objects": None,
    "People Only": {"person"},
    "Vehicles Only": {"car", "bus", "truck", "motorcycle", "bicycle"},
}


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


def process_video(
    input_path: str | Path,
    output_dir: str | Path,
    model_weights: str = "yolov8s.pt",
    conf_threshold: float = 0.60,
    iou_threshold: float = 0.60,
    detection_mode: str = "All Objects",
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> ProcessingResult:
    """
    Process *input_path* frame-by-frame, annotate detections, write output video.

    Processes every frame through YOLOv8 + ByteTrack (no frame skipping).
    Validates object persistence across frames to eliminate single-frame false positives.

    Parameters
    ----------
    input_path:
        Path to the source video file.
    output_dir:
        Directory where ``processed_<name>.mp4`` and ``detections.csv`` are saved.
    model_weights:
        YOLOv8 weights identifier.
    conf_threshold:
        Minimum detection confidence (0–1). Default 0.60.
    iou_threshold:
        IoU threshold for NMS (0–1). Default 0.60.
    progress_callback:
        Optional callable invoked each frame with (progress_fraction, current_frame,
        total_frames).  Useful for Streamlit progress bars.

    Returns
    -------
    ProcessingResult instance (check ``.error`` field first).
    """
    result = ProcessingResult()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    # ------------------------------------------------------------------ #
    # Validate                                                              #
    # ------------------------------------------------------------------ #
    validation_error = validate_video(input_path)
    if validation_error:
        result.error = validation_error
        _log.error("Validation failed: %s", validation_error)
        return result

    # ------------------------------------------------------------------ #
    # Load model                                                            #
    # ------------------------------------------------------------------ #
    try:
        model = load_model(model_weights)
    except Exception as exc:
        result.error = f"Failed to load model '{model_weights}': {exc}"
        _log.exception("Model load error")
        return result

    # ------------------------------------------------------------------ #
    # Open source video                                                     #
    # ------------------------------------------------------------------ #
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        result.error = f"Cannot open video: {input_path.name}"
        return result

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result.fps_input = fps
        result.resolution = f"{width}x{height}"

        _log.info(
            "Video opened: %s | %dx%d | %.1f FPS | %d frames",
            input_path.name, width, height, fps, total_frames,
        )

        # ---------------------------------------------------------------- #
        # Prepare output video writer with fourcc fallback                #
        # ---------------------------------------------------------------- #
        output_stem = f"processed_{input_path.stem}.mp4"
        output_video_path = output_dir / output_stem

        writer = None
        for code in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*code)
            w = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            if w.isOpened():
                writer = w
                _log.info("Opened VideoWriter with fourcc '%s'", code)
                break

        if writer is None or not writer.isOpened():
            result.error = (
                f"Cannot create output video at {output_video_path}. "
                "Check write permissions for the outputs directory."
            )
            return result

        # ---------------------------------------------------------------- #
        # Single-pass: detect → validate persistence → annotate → write    #
        #                                                                    #
        # Track IDs are promoted to 'validated' the instant they reach     #
        # min_persistence appearances.  No second video read required.      #
        # ---------------------------------------------------------------- #
        start_time = time.perf_counter()
        frame_idx = 0

        # ── Per-track quality state ──────────────────────────────────────
        # track_consecutive : unbroken run of consecutive frames for each ID
        # track_last_seen   : last frame_idx the ID appeared (gap detection)
        # track_class_history: all YOLO class predictions (majority-vote source)
        # track_conf_history : rolling window of confidences (stability gate)
        track_consecutive: Dict[int, int] = {}
        track_last_seen: Dict[int, int] = {}
        track_class_history: Dict[int, List[str]] = {}
        track_conf_history: Dict[int, List[float]] = {}
        validated_track_ids: Set[int] = set()

        _MIN_CONSECUTIVE = 2      # frames must appear without a gap to be valid
        _CONF_HIST_LEN   = 4      # rolling window size for variance check
        _MAX_CONF_VAR    = 0.015  # variance > this → erratic detection (stdev ≈ 0.12)
        _progress_every  = 5      # throttle Streamlit progress calls

        allowed_classes = MODE_ALLOWED_CLASSES.get(detection_mode, None)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break  # end of video

                # Run YOLO + ByteTrack on this frame
                detections: List[Detection] = detect_frame(
                    model, frame, frame_idx, conf_threshold, iou_threshold, allowed_classes=allowed_classes
                )

                # ── Stage 1: update per-track quality state ──────────────
                for det in detections:
                    tid = det.track_id

                    # Class history — always record for majority voting
                    cls_hist = track_class_history.setdefault(tid, [])
                    cls_hist.append(det.class_name)

                    # Confidence history — rolling window
                    conf_hist = track_conf_history.setdefault(tid, [])
                    conf_hist.append(det.confidence)
                    if len(conf_hist) > _CONF_HIST_LEN:
                        del conf_hist[0]

                    # Stage 2: confidence stability gate
                    # If variance of recent confidences is high the detection is
                    # erratic — reset the consecutive streak and skip validation.
                    if len(conf_hist) >= 3:
                        mean_c = sum(conf_hist) / len(conf_hist)
                        variance = sum((c - mean_c) ** 2 for c in conf_hist) / len(conf_hist)
                        if variance > _MAX_CONF_VAR:
                            track_consecutive[tid] = 0
                            track_last_seen[tid] = -2  # ensure next frame is not consecutive
                            continue

                    # Stage 3: consecutive-frame streak
                    # A gap of even one frame resets the streak to 1.
                    if frame_idx - track_last_seen.get(tid, -2) == 1:
                        track_consecutive[tid] = track_consecutive.get(tid, 0) + 1
                    else:
                        track_consecutive[tid] = 1
                    track_last_seen[tid] = frame_idx

                    # Promote to validated once streak reaches threshold
                    if track_consecutive[tid] >= _MIN_CONSECUTIVE:
                        validated_track_ids.add(tid)

                # Keep only detections with a validated (stable, persistent) track ID
                valid_dets = [
                    d for d in detections if d.track_id in validated_track_ids
                ]

                # Stage 4: majority-class override
                # Replace the per-frame YOLO class with the most-frequently predicted
                # class across ALL frames for this track ID.  This suppresses random
                # one-frame class switches (e.g. Car → Train → Car → Car → Car).
                for det in valid_dets:
                    hist = track_class_history.get(det.track_id)
                    if hist:
                        det.class_name = Counter(hist).most_common(1)[0][0]

                # Accumulate statistics for validated detections
                for det in valid_dets:
                    result.total_detections += 1
                    result.confidence_sum += det.confidence
                    result.all_records.append(det.to_dict())

                    cls = det.class_name
                    if cls not in result.unique_track_ids:
                        result.unique_track_ids[cls] = set()
                    result.unique_track_ids[cls].add(det.track_id)

                # Annotate in-place (we own `frame` exclusively at this point)
                # and write to output video.
                annotated = annotate_frame(frame, valid_dets, in_place=True)
                writer.write(annotated)

                frame_idx += 1
                result.frames_processed = frame_idx

                # Throttle progress updates to reduce Python/Streamlit overhead
                if progress_callback is not None:
                    if frame_idx % _progress_every == 0 or frame_idx == total_frames:
                        fraction = frame_idx / max(total_frames, 1)
                        progress_callback(min(fraction, 1.0), frame_idx, total_frames)

        finally:
            writer.release()

        # ---------------------------------------------------------------- #
        # Finalise statistics                                                #
        # ---------------------------------------------------------------- #
        result.processing_time = time.perf_counter() - start_time
        result.output_video_path = output_video_path

        _log.info(
            "Single-pass complete: %d frames in %s | %d track IDs seen | %d validated stable objects",
            result.frames_processed,
            format_seconds(result.processing_time),
            len(track_consecutive),
            result.unique_objects_count,
        )

        # ---------------------------------------------------------------- #
        # Export CSV                                                         #
        # ---------------------------------------------------------------- #
        csv_path = output_dir / "detections.csv"
        result.csv_path = export_detections_csv(result.all_records, csv_path)

        # ---------------------------------------------------------------- #
        # Generate AI summary                                                #
        # ---------------------------------------------------------------- #
        result.ai_summary = generate_ai_summary(result.to_stats_dict())

    except Exception as exc:
        result.error = f"Unexpected error during processing: {exc}"
        _log.exception("Processing pipeline error")

    finally:
        cap.release()

    return result
