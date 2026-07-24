"""
app.py – VisionGuard AI Streamlit application entry point.

Run with:
    streamlit run app.py

This module owns:
- Page configuration and NVIDIA-inspired dark CSS theme
- Upload section
- Processing trigger and progress reporting
- Results dashboard (statistics cards, detection table, charts)
- AI summary panel
- Download buttons
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so sibling modules are importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils import ensure_dirs, format_seconds, format_size_mb, get_logger
from video_processor import (
    SUPPORTED_EXTENSIONS,
    ProcessingResult,
    process_video,
    validate_video,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

UPLOAD_DIR = _PROJECT_ROOT / "uploads"
OUTPUT_DIR = _PROJECT_ROOT / "outputs"
ensure_dirs(UPLOAD_DIR, OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "VisionGuard AI – Professional Video Intelligence Platform",
    },
)

# ---------------------------------------------------------------------------
# NVIDIA-inspired dark CSS theme
# ---------------------------------------------------------------------------

DARK_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root palette ────────────────────────────────────────────────────── */
:root {
    --bg-primary:   #0a0a0f;
    --bg-secondary: #111118;
    --bg-card:      #16161f;
    --bg-card-2:    #1c1c27;
    --accent:       #76b900;   /* NVIDIA green */
    --accent-dim:   #4a7a00;
    --accent-glow:  rgba(118,185,0,0.18);
    --text-primary: #e8e8f0;
    --text-secondary:#9898b0;
    --text-muted:   #55556a;
    --border:       #2a2a3a;
    --border-accent:#76b900;
    --danger:       #ff4b6e;
    --warning:      #ffb800;
    --info:         #00b4d8;
    --radius:       12px;
    --radius-lg:    18px;
}

/* ── Global resets ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Main container */
.main .block-container {
    padding: 1.5rem 2rem 4rem;
    max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }

/* ── Hero header ─────────────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #111118 0%, #16161f 50%, #0a1200 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at top right, var(--accent-glow), transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #76b900, #b8ff00, #76b900);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin: 0 0 0.2rem;
}
.hero-subtitle {
    font-size: 1.0rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: var(--accent-glow);
    border: 1px solid var(--accent-dim);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

/* ── Card ────────────────────────────────────────────────────────────── */
.vg-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.vg-card:hover { border-color: var(--accent-dim); }

/* ── Stat card (metric) ──────────────────────────────────────────────── */
.stat-card {
    background: var(--bg-card-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: all 0.2s;
}
.stat-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 18px var(--accent-glow);
    transform: translateY(-2px);
}
.stat-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.25rem;
}

/* ── Section title ───────────────────────────────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.6rem 0 1rem;
}

/* ── AI summary box ──────────────────────────────────────────────────── */
.ai-summary {
    background: #0d1500;
    border: 1px solid var(--accent-dim);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #c8ff80;
    line-height: 1.7;
    white-space: pre;
    overflow-x: auto;
}

/* ── Upload zone ─────────────────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.25s, background 0.25s;
}
.upload-zone:hover {
    border-color: var(--accent);
    background: var(--accent-glow);
}

/* ── Streamlit component overrides ───────────────────────────────────── */
div[data-testid="stFileUploader"] > label { display: none; }
div[data-testid="stFileUploader"] section {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
}
div[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
    background: rgba(118,185,0,0.05) !important;
}

/* Progress bar */
div[data-testid="stProgressBar"] > div { background: var(--bg-card-2); }
div[data-testid="stProgressBar"] > div > div { background: var(--accent) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-dim)) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: var(--bg-card-2) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-dim) !important;
    font-weight: 600 !important;
    border-radius: var(--radius) !important;
}
.stDownloadButton > button:hover {
    background: var(--accent-glow) !important;
    border-color: var(--accent) !important;
}

/* Tables */
.stDataFrame { border-radius: var(--radius) !important; overflow: hidden; }
.stDataFrame [data-testid="stDataFrameResizable"] { border-radius: var(--radius); }

/* Alerts */
.stAlert { border-radius: var(--radius) !important; }

/* Selectbox / slider */
.stSelectbox > div > div,
.stSlider > div { border-radius: var(--radius) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state["result"] = None
if "processing" not in st.session_state:
    st.session_state["processing"] = False


# ---------------------------------------------------------------------------
# Helper: render a stat card
# ---------------------------------------------------------------------------

def stat_card(icon: str, value: str, label: str) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-icon">{icon}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Helper: Matplotlib bar chart  →  PNG bytes
# ---------------------------------------------------------------------------

def render_bar_chart(object_counts: Dict[str, int]) -> bytes:
    """
    Render a dark-themed bar chart of object counts and return PNG bytes.
    """
    names = list(object_counts.keys())
    counts = [object_counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    fig.patch.set_facecolor("#111118")
    ax.set_facecolor("#16161f")

    colours = ["#76b900" if i % 2 == 0 else "#4a7a00" for i in range(len(names))]
    bars = ax.bar(names, counts, color=colours, edgecolor="#2a2a3a", linewidth=0.8,
                  width=0.55)

    # Value annotations
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.05,
            str(count),
            ha="center",
            va="bottom",
            color="#e8e8f0",
            fontsize=10,
            fontweight="bold",
            fontfamily="monospace",
        )

    ax.set_xlabel("Object Class", color="#9898b0", fontsize=10)
    ax.set_ylabel("Unique Count", color="#9898b0", fontsize=10)
    ax.set_title("Detected Objects – Unique Counts", color="#e8e8f0", fontsize=12,
                 fontweight="bold", pad=14)
    ax.tick_params(colors="#9898b0", labelsize=9)
    ax.spines["bottom"].set_color("#2a2a3a")
    ax.spines["left"].set_color("#2a2a3a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#1e1e2a", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35 if len(names) > 6 else 0,
                        ha="right" if len(names) > 6 else "center",
                        color="#e8e8f0", fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="hero-badge">⚙ Configuration</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### VisionGuard AI")
    st.markdown(
        '<span style="color:#9898b0;font-size:0.82rem;">YOLOv8 + ByteTrack engine</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Minimum confidence score to keep a detection. Higher = fewer false positives.",
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Intersection-over-Union threshold for non-max suppression.",
    )

    st.divider()
    st.markdown("**Detection Mode**")
    detection_mode = st.selectbox(
        "Detection Mode",
        options=["All Objects", "People Only", "Vehicles Only"],
        index=0,
        help="Select object categories to detect and track.",
    )

    st.divider()
    st.markdown("**Model**")
    model_weights = st.selectbox(
        "YOLOv8 Weights",
        options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=1,
        help="Larger models are slower but more accurate.",
    )

    st.divider()
    st.markdown(
        '<span style="color:#55556a;font-size:0.75rem;">'
        "VisionGuard AI v1.0.0<br>"
        "YOLOv8 · ByteTrack · OpenCV<br>"
        "© 2025"
        "</span>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-badge">🛡️ AI POWERED</div>
        <div class="hero-title">VisionGuard AI</div>
        <div class="hero-subtitle">
            Professional Video Intelligence Platform &nbsp;·&nbsp;
            YOLOv8 Object Detection &nbsp;·&nbsp; ByteTrack Multi-Object Tracking
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">📂 Upload Video</div>', unsafe_allow_html=True)

col_up, col_info = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader(
        label="Drop video here",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, MOV",
        key="video_uploader",
    )

with col_info:
    st.markdown(
        """
        <div class="vg-card" style="height:100%;">
            <div style="color:#9898b0;font-size:0.82rem;line-height:1.9;">
                ✅ &nbsp;<strong>MP4</strong> &nbsp;·&nbsp; <strong>AVI</strong>
                &nbsp;·&nbsp; <strong>MOV</strong><br>
                🎯 &nbsp;YOLOv8n – ByteTrack<br>
                🔢 &nbsp;All COCO classes (80+)<br>
                🆔 &nbsp;Unique object tracking<br>
                📊 &nbsp;Auto-generated analytics<br>
                📥 &nbsp;CSV &amp; video download
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# File info + validation
# ---------------------------------------------------------------------------

saved_path: Optional[Path] = None

if uploaded_file is not None:
    file_size_mb = format_size_mb(uploaded_file.size)
    ext = Path(uploaded_file.name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        st.error(
            f"❌ Unsupported format **{ext}**. "
            f"Please upload {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    else:
        st.markdown(
            f"""
            <div class="vg-card">
                <span style="color:#9898b0;font-size:0.8rem;">📎 FILE INFO</span><br>
                <strong style="color:#e8e8f0;">{uploaded_file.name}</strong>
                &nbsp;&nbsp;
                <span style="color:#76b900;font-size:0.85rem;">({file_size_mb})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Save to uploads/
        saved_path = UPLOAD_DIR / uploaded_file.name
        saved_path.write_bytes(uploaded_file.getbuffer())

        # Quick pre-validation
        err = validate_video(saved_path)
        if err:
            st.error(f"❌ {err}")
            saved_path = None

# ---------------------------------------------------------------------------
# Process button
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">⚡ Processing</div>', unsafe_allow_html=True)

process_clicked = st.button(
    "🚀 Run VisionGuard AI",
    disabled=(saved_path is None or st.session_state["processing"]),
    use_container_width=True,
    key="run_btn",
)

if process_clicked and saved_path is not None:
    st.session_state["processing"] = True
    st.session_state["result"] = None

    # ── Progress UI ─────────────────────────────────────────────────────
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    with progress_placeholder.container():
        prog_bar = st.progress(0.0, text="Initialising…")

    def _progress_cb(fraction: float, current: int, total: int) -> None:
        pct = int(fraction * 100)
        prog_bar.progress(
            fraction,
            text=f"Processing frame {current} / {total} &nbsp; ({pct}%)",
        )

    with status_placeholder:
        with st.spinner("VisionGuard AI is processing your video…"):
            result: ProcessingResult = process_video(
                input_path=saved_path,
                output_dir=OUTPUT_DIR,
                model_weights=model_weights,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                detection_mode=detection_mode,
                progress_callback=_progress_cb,
            )

    prog_bar.progress(1.0, text="✅ Processing complete!")
    time.sleep(0.5)
    progress_placeholder.empty()
    status_placeholder.empty()

    st.session_state["result"] = result
    st.session_state["processing"] = False

    if result.error:
        st.error(f"❌ {result.error}")
    else:
        st.success("✅ Video processed successfully!")
        st.rerun()

# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------

result: Optional[ProcessingResult] = st.session_state.get("result")

if result is not None and result.error is None:

    # ── Stat cards ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">📊 Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    most_freq_raw = result.most_frequent_object
    most_freq_disp = most_freq_raw if most_freq_raw == "N/A" else most_freq_raw.capitalize()

    with c1:
        st.markdown(
            stat_card("🎞️", str(result.total_frames), "Frames Processed"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            stat_card("⏱️", format_seconds(result.processing_time), "Processing Time"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            stat_card("📐", result.resolution, "Resolution"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            stat_card("🎬", f"{result.fps:.1f}", "Source FPS"),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            stat_card("🔍", str(result.objects_detected), "Total Detections"),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            stat_card("🆔", str(result.unique_objects), "Unique Objects"),
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown(
            stat_card(
                "🎯",
                f"{result.average_confidence * 100:.1f}%",
                "Avg Confidence",
            ),
            unsafe_allow_html=True,
        )
    with c8:
        st.markdown(
            stat_card("🏆", most_freq_disp, "Most Frequent"),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Two-column layout: table + chart ────────────────────────────────
    col_table, col_chart = st.columns([1, 1], gap="large")

    with col_table:
        st.markdown(
            '<div class="section-title">🗂️ Detected Objects</div>',
            unsafe_allow_html=True,
        )

        obj_counts = result.object_counts
        if obj_counts:
            df_obj = pd.DataFrame(
                [
                    {"Object": k.capitalize(), "Unique Count": v}
                    for k, v in sorted(obj_counts.items(), key=lambda x: -x[1])
                ]
            )
            st.dataframe(
                df_obj,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Object": st.column_config.TextColumn("Object", width="medium"),
                    "Unique Count": st.column_config.NumberColumn(
                        "Unique Count",
                        format="%d",
                        width="small",
                    ),
                },
            )
        else:
            st.info("No objects detected in this video.")

    with col_chart:
        st.markdown(
            '<div class="section-title">📈 Object Distribution</div>',
            unsafe_allow_html=True,
        )

        if obj_counts:
            chart_bytes = render_bar_chart(obj_counts)
            st.image(chart_bytes, use_container_width=True)
        else:
            st.info("No objects detected in this video.")

    st.divider()

    # ── Full detections table ────────────────────────────────────────────
    with st.expander("🔎 Full Detection Records", expanded=False):
        if result.detections:
            df_full = pd.DataFrame(result.detections)
            df_full.columns = [c.replace("_", " ").title() for c in df_full.columns]
            st.dataframe(df_full, use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected in this video.")

    st.divider()

    # ── Video preview + download ─────────────────────────────────────────
    col_vid, col_dl = st.columns([3, 1], gap="large")

    with col_vid:
        st.markdown(
            '<div class="section-title">🎥 Processed Video Preview</div>',
            unsafe_allow_html=True,
        )
        if (
            result.output_path
            and result.output_path.exists()
            and result.output_path.stat().st_size > 0
        ):
            try:
                video_bytes = result.output_path.read_bytes()
                st.video(video_bytes, format="video/mp4")
            except Exception as exc:
                _log.warning("Streamlit st.video preview error: %s", exc)
                st.warning("Video preview unavailable. Please download the processed video.")
        else:
            st.warning("Video preview unavailable. Please download the processed video.")

    with col_dl:
        st.markdown(
            '<div class="section-title">📥 Downloads</div>',
            unsafe_allow_html=True,
        )

        if (
            result.output_path
            and result.output_path.exists()
            and result.output_path.stat().st_size > 0
        ):
            st.download_button(
                label="⬇️ Download Processed Video",
                data=result.output_path.read_bytes(),
                file_name=result.output_path.name,
                mime="video/mp4",
                use_container_width=True,
                key="dl_video",
            )
        else:
            st.button("⬇️ Download Video (unavailable)", disabled=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if result.csv_path and result.csv_path.exists() and result.csv_path.stat().st_size > 0:
            st.download_button(
                label="⬇️ Download detections.csv",
                data=result.csv_path.read_bytes(),
                file_name="detections.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv",
            )
        else:
            st.button("⬇️ Download CSV (unavailable)", disabled=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if result.ai_summary:
            summary_bytes = result.ai_summary.encode("utf-8")
            st.download_button(
                label="⬇️ Download AI Summary",
                data=summary_bytes,
                file_name="ai_summary.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_summary",
            )

    st.divider()

    # ── AI summary ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">🤖 AI Video Summary</div>',
        unsafe_allow_html=True,
    )

    if result.ai_summary:
        st.markdown(
            f'<div class="ai-summary">{result.ai_summary}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No objects detected in this video.")

elif result is not None and result.error:
    # Show error state
    st.markdown(
        '<div class="section-title">❌ Processing Error</div>',
        unsafe_allow_html=True,
    )
    st.error(result.error)

else:
    # Landing / idle state
    st.markdown(
        """
        <div class="vg-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">🛡️</div>
            <div style="font-size:1.3rem;font-weight:700;color:#e8e8f0;margin-bottom:0.5rem;">
                Ready for Analysis
            </div>
            <div style="color:#9898b0;font-size:0.9rem;max-width:480px;margin:0 auto;">
                Upload an MP4, AVI, or MOV video above, then click
                <strong style="color:#76b900;">Run VisionGuard AI</strong> to start
                real-time object detection and tracking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
