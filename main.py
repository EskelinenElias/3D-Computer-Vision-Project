import json

import cv2
import numpy as np
from PIL import Image

from camera_calibration_dlt import calibrate as _calibrate_dlt
from intrinsics import calibrate_intrinsics
from extrinsics import solve_scene_pose
from object_detection import _detect_cubes, _detect_target_locations, _detect_robot
from robot_control import _plan_for_color, _translate


PATTERN_SIZE = (8, 6)
SQUARE_SIZE_CM = 4.0


def calibrate(intrinsic_imgs, extrinsic_img=None, method="zhang", **kwargs):
    """Calibrate the camera for a scene.

    The calibration is a two-step process:
      1. Intrinsics (K) — a property of the lens + sensor, recoverable once
         per camera. Uses `calibrate_intrinsics` for method="zhang", or the
         DLT decomposition for method="dlt".
      2. Extrinsics (R_scene, t_scene) — where the camera is in the world
         for this specific scene. Uses `solve_scene_pose` (solvePnP on the
         checkerboard) for method="zhang", or DLT's built-in pose output.

    Args:
        intrinsic_imgs (list<PIL.Image>): images used to recover K.
            - method="zhang": pose-diverse checkerboard images.
            - method="dlt":   single-element list [dlt_image].
        extrinsic_img (PIL.Image, optional): scene-pose reference image
            (checkerboard at its world-frame location, camera at the scene's
            pose). If None:
              - Zhang: falls back to `intrinsic_imgs[-1]`. Only correct when
                the camera didn't move between calibration and the scene.
              - DLT:   falls back to `intrinsic_imgs[0]` (the DLT image).
        method (str): "zhang" (default) or "dlt".
        **kwargs:
            - method="dlt": must supply `points_2d` + `points_3d` (np.ndarray)
              or `dlt_points_file` (JSON path with "points_2d"/"points_3d").

    Returns:
        dict with 'K', 'dist', 'image_size', 'R_scene', 't_scene',
        'extrinsic_rms' (scene-pose reprojection error in pixels).
    """
    if method not in ("zhang", "dlt"):
        raise ValueError(f"Unknown method: {method!r} (expected 'zhang' or 'dlt')")

    if extrinsic_img is None:
        extrinsic_img = intrinsic_imgs[-1 if method == "zhang" else 0]

    if method == "zhang":
        K = calibrate_intrinsics(intrinsic_imgs, method="zhang")
        R_scene, t_scene, reproj_rms = solve_scene_pose(extrinsic_img, K)
    else:
        if "dlt_points_file" in kwargs:
            with open(kwargs.pop("dlt_points_file")) as f:
                saved = json.load(f)
            kwargs["points_2d"] = np.asarray(saved["points_2d"], dtype=np.float64)
            kwargs["points_3d"] = np.asarray(saved["points_3d"], dtype=np.float64)

        dlt_result = _calibrate_dlt(
            extrinsic_img,
            points_2d=kwargs.get("points_2d"),
            points_3d=kwargs.get("points_3d"),
        )
        K = dlt_result["K"]
        R_scene, _ = cv2.Rodrigues(dlt_result["rvecs"][-1])
        t_scene = dlt_result["tvecs"][-1].ravel()
        reproj_rms = None

    if reproj_rms is not None:
        print(f"[calibrate] scene-pose RMS: {reproj_rms:.3f} px")

    return {
        "K": K,
        "dist": np.zeros((1, 5)),
        "image_size": extrinsic_img.size,
        "R_scene": R_scene,
        "t_scene": t_scene,
        "extrinsic_rms": reproj_rms,
    }


def move_block(blocks, img, calib, toggle_dir: bool=False):
    """Return the command string to pick up and place the given cubes.

    Args:
        blocks (list<str>): ordered list of cube colours to move.
        img (PIL.Image): scene image with robot and cubes visible.
        calib (dict): output of `calibrate`.

    Returns:
        str: ";" -separated commands, e.g.
             "turn(45.0); go(20.0); grab(); turn(-30.0); go(15.0); let_go()"
    """
    R, t = calib["R_scene"], calib["t_scene"]

    hsv = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2HSV)
    cubes   = _detect_cubes(hsv, calib, R, t)
    targets = _detect_target_locations(hsv, calib, R, t)
    robot   = _detect_robot(hsv, calib, R, t)

    robot_state = {"pos": robot["pos"][:2].copy(), "heading": robot["heading"]}
    all_steps = []
    for color in blocks:
        try:
            all_steps.extend(_plan_for_color(color, robot_state, cubes, targets))
        except KeyError as e:
            print(f"Skipping {color}: {e}")

    return _translate(all_steps, toggle_dir)


if __name__ == "__main__":

    from pathlib import Path

    INTRINSIC_DIR = Path("test-images/intrinsic_calibration")
    SCENE_DIR     = Path("test-images/scene6")
    BLOCKS = ["red", "green", "blue"]

    intrinsic_imgs = [Image.open(p) for p in sorted(INTRINSIC_DIR.glob("*.png"))]
    extrinsic_img  = Image.open(sorted((SCENE_DIR / "calibration").glob("*.png"))[0])
    scene_img      = Image.open(sorted((SCENE_DIR / "images").glob("*.png"))[0])

    # Explicit two-arg call — K calibrated from intrinsic_imgs, scene pose
    # recovered from extrinsic_img (different camera pose is fine).
    calib = calibrate(intrinsic_imgs, extrinsic_img, method="zhang")
    K = calib["K"]
    print(f"K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
          f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    commands = move_block(BLOCKS, scene_img, calib)
    print(f"\nCommands for blocks={BLOCKS}:")
    print(commands)

    assert isinstance(commands, str)
    assert commands.count("grab()") == commands.count("let_go()")
    print("\nAll sanity checks passed.")
