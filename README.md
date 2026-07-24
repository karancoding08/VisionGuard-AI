# VisionGuard AI

> **Professional Video Intelligence Platform** powered by YOLOv8 + ByteTrack

---

## Overview

VisionGuard AI is a production-ready, NVIDIA-inspired Streamlit application that processes
uploaded videos with real-time multi-object detection and tracking.

- **Detection**: YOLOv8n (or larger variants) – covers all 80 COCO classes automatically
- **Tracking**: ByteTrack – assigns persistent IDs so the same person across 300 frames
  counts as **one** unique person
- **Output**: annotated video, CSV of every detection, bar chart, and AI summary

---

## Features

| Feature | Details |
|---|---|
| 📤 Upload | MP4, AVI, MOV – validated before processing |
| 🎯 Detection | All COCO 80 classes (person, chair, car, dog, …) – no hardcoding |
| 🆔 Tracking | ByteTrack – unique ID per physical object across all frames |
| 🎬 Annotated video | Coloured bounding boxes + `Class \| ID n \| conf%` labels |
| 📊 Dashboard | Frames, Time, FPS, Resolution, Detections, Unique Objects, Avg Conf |
| 🗂️ Detected Objects table | Auto-generated from results |
| 📈 Bar chart | Matplotlib – Object vs Unique Count |
| 🤖 AI Summary | Auto-generated text block |
| 📥 Downloads | Processed video + `detections.csv` + `ai_summary.txt` |
| ⚙️ Config sidebar | Confidence & IoU sliders, model weight selector |

---

## Project Structure

```
VisionGuard-AI/
├── app.py               # Streamlit entry point + UI
├── detector.py          # YOLOv8 model loader, frame inference, annotation
├── video_processor.py   # End-to-end pipeline (read → detect → write → stats)
├── utils.py             # Colours, formatting, CSV export, AI summary
├── requirements.txt
├── README.md
├── uploads/             # Temporary uploaded videos (auto-created)
└── outputs/             # Processed videos + CSV (auto-created)
```

---

## Installation

### 1. Clone / copy the project

```bash
# already in the project folder
cd VisionGuard-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `ultralytics` will automatically download `yolov8n.pt` on first run
> (~6 MB). ByteTrack is bundled with Ultralytics – no separate installation needed.
> `lapx` is required by ByteTrack for the Hungarian algorithm.

---

## Usage

### Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Workflow

1. Upload a video (MP4 / AVI / MOV) via the **Upload Video** section.
2. Optionally tune **Confidence Threshold**, **IoU Threshold**, and **Model** in the sidebar.
3. Click **🚀 Run VisionGuard AI**.
4. Watch the animated progress bar.
5. Review the analytics dashboard, detection table, and bar chart.
6. Watch or download the annotated video.
7. Download `detections.csv` or the AI Summary text.

---

## Configuration (Sidebar)

| Parameter | Default | Range | Effect |
|---|---|---|---|
| Confidence Threshold | 0.35 | 0.10 – 0.95 | Filters low-confidence detections |
| IoU Threshold (NMS) | 0.45 | 0.10 – 0.90 | Controls overlap suppression |
| YOLOv8 Weights | yolov8n.pt | n / s / m / l / x | Speed vs accuracy trade-off |

---

## Output Files

| File | Location | Contents |
|---|---|---|
| `processed_<name>.mp4` | `outputs/` | Annotated video |
| `detections.csv` | `outputs/` | Every detection (frame, track\_id, object, confidence, bbox) |
| `ai_summary.txt` | download only | Text summary block |

### `detections.csv` columns

```
frame, track_id, object, confidence, bbox
0,     1,        person, 0.9823,     (120,30,280,450)
0,     2,        chair,  0.8741,     (300,200,450,400)
...
```

---

## YOLO Classes Supported (automatic – no code changes needed)

All 80 COCO classes are supported out of the box:

`person · bicycle · car · motorcycle · airplane · bus · train · truck · boat ·
traffic light · fire hydrant · stop sign · parking meter · bench · bird · cat ·
dog · horse · sheep · cow · elephant · bear · zebra · giraffe · backpack ·
umbrella · handbag · tie · suitcase · frisbee · skis · snowboard · sports ball ·
kite · baseball bat · baseball glove · skateboard · surfboard · tennis racket ·
bottle · wine glass · cup · fork · knife · spoon · bowl · banana · apple ·
sandwich · orange · broccoli · carrot · hot dog · pizza · donut · cake · chair ·
couch · potted plant · bed · dining table · toilet · tv · laptop · mouse ·
remote · keyboard · cell phone · microwave · oven · toaster · sink ·
refrigerator · book · clock · vase · scissors · teddy bear · hair drier · toothbrush`

---

## System Requirements

- Python 3.11+
- Windows 10/11 (also works on Linux/macOS)
- 4 GB RAM minimum (8 GB recommended for larger models)
- NVIDIA GPU optional – CUDA speeds up inference significantly

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: lapx` | `pip install lapx` |
| `No module named 'ultralytics'` | `pip install ultralytics` |
| `FileNotFoundError: yolov8n.pt` | Run with internet access once so Ultralytics downloads it |
| Output video is blank | Ensure `outputs/` directory is writable |
| Progress bar freezes | Normal for very large videos – ByteTrack has slight overhead |

---

## License

MIT License – free to use, modify, and distribute.
