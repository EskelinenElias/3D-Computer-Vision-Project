# 3D Computer Vision Project

Camera calibration, object detection, and pick-and-place motion planning from a single overhead image.

**Authors:** Jarkko Komulainen, Matti Aalto, Elias Eskelinen

## Pipeline

1. **Checkerboard detection** — [checkerboard_detection.py](checkerboard_detection.py) — OpenCV's SB detector with sub-pixel refinement.
2. **Camera calibration** — [camera_calibration.py](camera_calibration.py) — Zhang's method (DLT homographies → closed-form `K` → per-view extrinsics → joint LM refinement), with EXIF fallback for degenerate view geometry.
3. **Object detection** — [object_detection.py](object_detection.py) — HSV segmentation + contour hierarchy to distinguish solid cubes from ring-shaped targets. Ray-plane intersection gives 3D world coordinates at each object's known height plane.
4. **Robot control** — [robot_control.py](robot_control.py) — pick-and-place planner producing structured step dicts; separate translator serialises them to the assignment's `cmd; cmd; ...` string format.
5. **End-to-end demo** — [demo.ipynb](demo.ipynb) — runs all four stages on the test images and animates the planned robot motion both top-down and overlaid on the scene image.

## Prerequisites

- Python **3.13** (specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) for environment + dependency management

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

Clone the repo, then sync dependencies:

```bash
git clone <repo-url>
cd 3D-Computer-Vision-Project
uv sync
```

`uv sync` creates `.venv/` with Python 3.13 and installs everything from `pyproject.toml` / `uv.lock`. No manual `pip install`, no `activate` required — `uv run ...` picks up the env automatically.

## Running

Each pipeline stage has a self-contained `__main__` block that calibrates, detects, and visualises using [test-images/set2/](test-images/):

```bash
uv run python checkerboard_detection.py   # corner detection
uv run python camera_calibration.py       # Zhang's calibration + reprojection viz
uv run python object_detection.py         # HSV detection + top-down plot
uv run python robot_control.py            # motion planning + animation
```

For the end-to-end walkthrough with inline animations:

```bash
uv run jupyter lab demo.ipynb
```

## Project layout

```text
.
├── checkerboard_detection.py   # 1. Corner detection
├── camera_calibration.py       # 2. Zhang's method + LM refinement
├── object_detection.py         # 3. HSV + contour hierarchy
├── robot_control.py            # 4. Pick-and-place planner + animation
├── assignment.py               # Assignment stub (calibrate/move_block signatures)
├── demo.ipynb                  # End-to-end notebook
├── test-images/
│   ├── set1/  (2 calibration + scene images)
│   └── set2/  (3 calibration + 14 scene images)
├── figures/                    # Generated plots (git-ignored)
├── pyproject.toml              # uv-managed project metadata
├── uv.lock                     # Pinned dependency versions
└── .python-version             # 3.13
```

## Sources

Z. Zhang (2000) *A Flexible New Technique for Camera Calibration.* IEEE Transactions on Pattern Analysis and Machine Intelligence 22(11): 1330–1334.
