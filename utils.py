"""
utils.py – Utility helpers for VisionGuard AI.

Provides:
  - Deterministic per-class colour generation
  - Human-readable formatting helpers
  - AI summary text generation
  - CSV export
  - Logging configuration
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Fixed high-contrast palette (BGR for OpenCV)
_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 255, 127),    # spring green
    (255, 128, 0),    # sky blue
    (0, 165, 255),    # orange
    (147, 20, 255),   # deep pink
    (0, 255, 255),    # yellow
    (255, 0, 128),    # violet
    (0, 215, 255),    # gold
    (128, 255, 0),    # lime
    (255, 64, 64),    # steel blue
    (60, 179, 113),   # medium sea green
    (255, 0, 255),    # magenta
    (0, 128, 255),    # dark orange
    (204, 51, 255),   # orchid
    (255, 215, 0),    # deep sky blue
    (0, 255, 200),    # medium spring green
    (255, 20, 147),   # hot pink (BGR)
]


def class_colour(class_name: str) -> Tuple[int, int, int]:
    """
    Return a deterministic BGR colour for *class_name*.

    The same class name always gets the same colour across frames.
    """
    idx = hash(class_name) % len(_PALETTE_BGR)
    return _PALETTE_BGR[idx]


def track_id_colour(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR colour based on *track_id*."""
    idx = int(track_id) % len(_PALETTE_BGR)
    return _PALETTE_BGR[idx]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_seconds(seconds: float) -> str:
    """Convert *seconds* to a human-readable string (e.g. '1 min 3 sec')."""
    if seconds < 60:
        return f"{int(seconds)} sec"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes} min {secs} sec"


def format_confidence(conf: float) -> str:
    """Format a confidence value (0-1) as a percentage string."""
    return f"{conf * 100:.1f}%"


def format_size_mb(size_bytes: int) -> str:
    """Return a human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.2f} MB"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_detections_csv(
    records: List[Dict],
    output_path: str | Path,
) -> Path:
    """
    Write detection records to a CSV file.

    Parameters
    ----------
    records:
        List of dicts with keys: frame, track_id, object, confidence, bbox.
    output_path:
        Destination file path.

    Returns
    -------
    Path to the written CSV file.
    """
    output_path = Path(output_path)
    ensure_dirs(output_path.parent)

    fieldnames = ["frame", "track_id", "object", "confidence", "bbox"]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "frame": rec.get("frame", ""),
                    "track_id": rec.get("track_id", ""),
                    "object": rec.get("object", ""),
                    "confidence": f"{rec.get('confidence', 0.0):.4f}",
                    "bbox": rec.get("bbox", ""),
                }
            )

    _log.info("CSV exported → %s  (%d rows)", output_path, len(records))
    return output_path


# ---------------------------------------------------------------------------
# AI summary generation
# ---------------------------------------------------------------------------

def generate_ai_summary(stats: Dict) -> str:
    """
    Build the AI Video Summary text block from *stats*.

    Expected keys in *stats*:
        frames_processed, processing_time, resolution, fps,
        total_detections, unique_objects_count, average_confidence,
        object_counts (dict[str, int]), most_frequent_object.
    """
    frames: int = stats.get("frames_processed", 0)
    proc_time: float = stats.get("processing_time", 0.0)
    resolution: str = stats.get("resolution", "N/A")
    fps: float = stats.get("fps", 0.0)
    total_det: int = stats.get("total_detections", 0)
    unique_obj: int = stats.get("unique_objects_count", 0)
    avg_conf: float = stats.get("average_confidence", 0.0)
    obj_counts: Dict[str, int] = stats.get("object_counts", {})
    most_freq: str = stats.get("most_frequent_object", "N/A")

    lines: List[str] = [
        "=" * 46,
        "          AI VIDEO SUMMARY",
        "=" * 46,
        "",
        f"  Frames Processed  : {frames}",
        f"  Processing Time   : {format_seconds(proc_time)}",
        f"  Resolution        : {resolution}",
        f"  FPS               : {fps:.1f}",
        f"  Total Detections  : {total_det}",
        f"  Unique Objects    : {unique_obj}",
        f"  Average Confidence: {format_confidence(avg_conf)}",
        "",
        "  Detected Objects",
    ]

    if obj_counts:
        for obj_name, count in sorted(obj_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {obj_name:<18}: {count}")
    else:
        lines.append("    No objects detected in this video.")

    lines += [
        "",
        f"  Most Frequent Object",
        f"    {most_freq}",
        "",
        "  Status",
        "    SUCCESS",
        "",
        "=" * 46,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator / denominator, or *fallback* if denominator is zero."""
    return numerator / denominator if denominator else fallback
