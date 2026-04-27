import json

import cv2
import numpy as np
from PIL import Image

from camera_calibration_dlt import calibrate_DLT
from camera_calibration_zhang import compute_intrinsics, compute_extrinsics
from object_detection import _detect_cubes, _detect_targets, _detect_robot
from robot_control import _plan_for_color, _translate
from DLT_calibration_boxes import calibrate_DLT_boxes
 

def calibrate(intrinsic_imgs, extrinsic_img=None, method="zhang", pattern_size=(8,6)):
    """Calibrate the camera for a scene.

    The calibration is a two-step process:
      1. Intrinsics (K) — a property of the lens + sensor, recoverable once
         per camera. Uses Zhang's method for method="zhang", or the DLT
         decomposition for method="dlt".
      2. Extrinsics (R_scene, t_scene) — where the camera is in the world
         for this specific scene. Uses solvePnP for method="zhang", or DLT's
         built-in pose output for method="dlt".

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
        method (str): "zhang" (default) or "dlt" or "dlt_boxes".
        pattern_size (tuple): size of the checkerboard pattern, default (8,6)

    Returns:
        dict with 'K', 'image_size', 'R_scene', 't_scene',
        'extrinsic_rms' (scene-pose reprojection error in pixels).
    """
    if method not in ("zhang", "dlt","dlt_boxes"):
        raise ValueError(f"Unknown method: {method!r} (expected 'zhang' or 'dlt')")

    if extrinsic_img is None:
        extrinsic_img = intrinsic_imgs[-1 if method == "zhang" else 0]

    if method == "zhang":
        K = compute_intrinsics(intrinsic_imgs, pattern_size)
        R_scene, t_scene, reproj_rms = compute_extrinsics(extrinsic_img, K, pattern_size)
    elif method == "dlt":
        _, K, R_scene, C = calibrate_DLT(extrinsic_img, pattern_size)
        t_scene = -R_scene @ C
        reproj_rms = None
    elif method == 'dlt_boxes':
        _, K, R_scene, C = calibrate_DLT_boxes(extrinsic_img)
        t_scene = -R_scene @ C
        reproj_rms = None

    if reproj_rms is not None:
        print(f"[compute_extrinsics] scene-pose RMS: {reproj_rms:.3f} px")
    

    return {
        "K": K,
        "image_size": extrinsic_img.size,
        "R_scene": R_scene,
        "t_scene": t_scene,
        "extrinsic_rms": reproj_rms,
    }


def move_block(blocks, img, calib,
               drive_scale=1.0, drive_bias=0.0,
               turn_scale=1.0, turn_bias=0.0):
    """Return the command string to pick up and place the given cubes.

    Args:
        blocks (list<str>): ordered list of cube colours to move.
        img (PIL.Image): scene image with robot and cubes visible.
        calib (dict): output of `calibrate`.
        drive_scale, drive_bias: real-world drive calibration; output is
            drive_scale * d + sign(d) * drive_bias.
        turn_scale, turn_bias: real-world turn calibration; output is
            turn_scale * theta + sign(theta) * turn_bias. Pass turn_scale=-1.0
            to flip turn direction.

    Returns:
        str: ";" -separated commands, e.g.
             "turn(45.0); go(20.0); grab(); turn(-30.0); go(15.0); let_go()"
    """
    K, R, t = calib["K"], calib["R_scene"], calib["t_scene"]

    hsv = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2HSV)
    cubes   = _detect_cubes(hsv, K, R, t)
    targets = _detect_targets(hsv, K, R, t)
    robot   = _detect_robot(hsv, K, R, t)

    robot_state = {"pos": robot["pos"][:2].copy(), "heading": robot["heading"]}
    all_steps = []
    for color in blocks:
        try:
            all_steps.extend(_plan_for_color(color, robot_state, cubes, targets))
        except KeyError as e:
            print(f"Skipping {color}: {e}")

    return _translate(all_steps,
                      drive_scale=drive_scale, drive_bias=drive_bias,
                      turn_scale=turn_scale, turn_bias=turn_bias)


if __name__ == "__main__":

    from pathlib import Path

    # Choose method
    method = 'dlt_boxes'#'zhang'

    if method == 'zhang':
        INTRINSIC_DIR = Path("test-images/intrinsic_calibration")
        SCENE_DIR     = Path("test-images/scene6")
        BLOCKS = ["red", "green", "blue"]
    elif method == 'dlt':
        INTRINSIC_DIR = Path("test-images/set1")
        SCENE_DIR     = Path("test-images/set1")
        BLOCKS = ["red", "green", "blue"]
    elif method == 'dlt_boxes':
        INTRINSIC_DIR = Path("test-images/dlt-set-4")
        SCENE_DIR     = Path("test-images/dlt-set-4")
        BLOCKS = ["green","red", "blue"]

    intrinsic_imgs = [Image.open(p) for p in sorted(INTRINSIC_DIR.glob("*.png"))]
    extrinsic_img  = Image.open(sorted((SCENE_DIR / "calibration").glob("*.png"))[0])
    scene_img      = Image.open(sorted((SCENE_DIR / "images").glob("*.png"))[0])
    
    # Perform calibration
    calib = calibrate(intrinsic_imgs, extrinsic_img, method=method)
   
    # Run object detection and generate commands
    commands = move_block(BLOCKS, scene_img, calib)
    print(f"\nCommands for blocks={BLOCKS}:")
    print(commands)