# 3D Computer Vision Project

Camera calibration, object detection, and pick-and-place motion planning from a single overhead image.

**Authors:** Elias Eskelinen, Jarkko Komulainen, Matti Aalto

## Pipeline

1. **Checkerboard detection** — [checkerboard_detection.py](checkerboard_detection.py) — OpenCV's SB detector with sub-pixel refinement.
2. **Intrinsic calibration** — [intrinsics.py](intrinsics.py) — recovers the camera's intrinsic matrix **K** (focal length, principal point). Two backends:
   - **Zhang** ([camera_calibration_zhang.py](camera_calibration_zhang.py)) — multiple checkerboard images at different board orientations; DLT homographies → closed-form K → joint LM refinement.
   - **DLT** ([camera_calibration_dlt.py](camera_calibration_dlt.py)) — single image + manual 2D↔3D correspondences; projection-matrix decomposition yields K, R, t together.
3. **Extrinsic calibration** — [extrinsics.py](extrinsics.py) — recovers the scene pose **(R, t)** for a given camera setup via OpenCV's `solvePnP` on the checkerboard. Runs once per scene.
4. **Object detection** — [object_detection.py](object_detection.py) — HSV segmentation + contour hierarchy to distinguish solid cubes from ring-shaped targets. Ray-plane intersection gives 3D world coordinates at each object's known height plane.
5. **Robot control** — [robot_control.py](robot_control.py) — pick-and-place planner producing structured step dicts; separate translator serialises them to the `cmd; cmd; ...` string format the robot expects.
6. **Orchestration** — [main.py](main.py) — exposes the assignment-level `calibrate(intrinsic_imgs, extrinsic_img, method="zhang"|"dlt")` and `move_block(blocks, scene_img, calib)` functions.

Two notebooks:

- [demo.ipynb](demo.ipynb) — minimal end-to-end demo that calls `main.calibrate` + `main.move_block`.
- [test.ipynb](test.ipynb) — full walkthrough with per-step visualisations and inline animations.

## Prerequisites

- Python **3.13** (specified in `.python-version`)
- [uv](https://docs.astral.sh/uv/) for environment + dependency management

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone <repo-url>
cd 3D-Computer-Vision-Project
uv sync
```

`uv sync` creates `.venv/` with Python 3.13 and installs everything from `pyproject.toml` / `uv.lock`. `uv run ...` picks up the env automatically — no manual activation needed.

## Running

Each pipeline file has a self-contained `__main__` block that demos the module on the included test images:

```bash
uv run python checkerboard_detection.py    # corner detection
uv run python intrinsics.py                # K via Zhang
uv run python extrinsics.py                # K + scene pose via solvePnP
uv run python camera_calibration_zhang.py  # full Zhang pipeline with viz
uv run python camera_calibration_dlt.py    # DLT backup (requires saved points)
uv run python object_detection.py          # HSV + contour detection + top-down plot
uv run python robot_control.py             # motion planning + animation
uv run python main.py                      # end-to-end: calibrate → detect → plan
```

The end-to-end notebook:

```bash
uv run jupyter lab demo.ipynb
```

## Typical usage

```python
from main import calibrate, move_block
from PIL import Image
from pathlib import Path

# Intrinsic images: checkerboard at several different orientations (one-time per camera)
intrinsic_imgs = [Image.open(p) for p in sorted(Path("test-images/intrinsic_calibration").glob("*.png"))]

# Extrinsic image: checkerboard visible in the scene's setup (one per scene arrangement)
extrinsic_img = Image.open("test-images/scene6/calibration/image0086.png")

# Calibrate (K + scene pose)
calib = calibrate(intrinsic_imgs, extrinsic_img, method="zhang")

# Detect and plan on the task image
scene_img = Image.open("test-images/scene6/images/image0087.png")
commands = move_block(["red", "green", "blue"], scene_img, calib)
print(commands)
# -> "turn(-124.9); go(10.0); grab(); turn(99.5); go(25.0); let_go(); ..."
```

### DLT backup pipeline

If Zhang's method fails (all calibration images share similar board orientation — the closed-form K extraction then becomes ill-conditioned), switch to DLT:

```python
calib = calibrate(
    intrinsic_imgs=[Image.open("test-images/set4/calibration/image0086.png")],
    method="dlt",
    dlt_points_file="test-images/set4/dlt_points.json",   # saved 2D↔3D points
)
```

DLT requires a single image with ≥6 non-coplanar 3D points at known world locations (e.g. cube corners at `z=4` + floor corners at `z=0`). Use `camera_calibration_dlt.select_points` to pick them interactively.

## Calibration strategy

Two image collections feed the pipeline:

- **`test-images/intrinsic_calibration/`** — checkerboard photographed from many different angles. Used once to compute K. For Zhang's closed-form to succeed, **the board's orientation must vary between views** — flat-on-the-table-from-different-camera-positions is *not* enough, because the per-view homographies become 2D-affine-equivalent and can't constrain the focal length. Tilt the board (rigid backing helps since paper curls) or swing the camera through significantly different pitch/yaw.
- **`test-images/scene<N>/calibration/`** — one image per scene, checkerboard lying at a known world origin on the robot's table, camera at the scene's working pose. Used for `solve_scene_pose`. Pose diversity doesn't matter here (single-image PnP).
- **`test-images/scene<N>/images/`** — the task images (cubes, robot, targets). Must be shot from the same camera pose as the scene's calibration image.

If Zhang's closed-form produces an implausible K (very different fx and fy, or principal point outside the image), the pipeline falls back to a default heuristic K + LM refinement. The fallback typically produces a usable calibration anyway, but the warning is a signal that your intrinsic images lack orientation diversity.

## Project layout

```text
.
├── main.py                       # Orchestration: calibrate + move_block
├── intrinsics.py                 # calibrate_intrinsics (K only)
├── extrinsics.py                 # solve_scene_pose (R, t via solvePnP)
├── camera_calibration_zhang.py   # Zhang's method (K + per-view extrinsics)
├── camera_calibration_dlt.py     # DLT: K + R + t from one image
├── checkerboard_detection.py     # Corner detection
├── object_detection.py           # Cubes/targets/robot in HSV + world coords
├── robot_control.py              # Pick-and-place planner + animation
├── demo.ipynb                    # Minimal end-to-end demo
├── test.ipynb                    # Full walkthrough with animations
├── test-images/
│   ├── intrinsic_calibration/    # Board at diverse orientations → K
│   └── scene6/
│       ├── calibration/          # Scene-pose reference image → (R, t)
│       └── images/               # Task images (robot, cubes, targets)
├── figures/                      # Generated plots
├── pyproject.toml                # uv-managed project metadata
├── uv.lock                       # Pinned dependency versions
└── .python-version               # 3.13
```

## Sources

Z. Zhang (2000) *A Flexible New Technique for Camera Calibration.* IEEE Transactions on Pattern Analysis and Machine Intelligence 22(11): 1330–1334.
