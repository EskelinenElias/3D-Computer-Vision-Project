"""
Extrinsic camera calibration — recover the scene pose (R, t) given a known K.

Runs once per scene (every time the camera is moved). Takes a single image
containing the checkerboard at a known world location, detects its corners,
and solves for the camera's pose via OpenCV's solvePnP.
"""

import cv2
import numpy as np

from checkerboard_detection import _find_checkerboard_corners
from camera_calibration_zhang import _build_object_points


PATTERN_SIZE = (8, 6)
SQUARE_SIZE_CM = 4.0


def solve_scene_pose(scene_img, K, pattern_size=PATTERN_SIZE,
                     square_size_cm=SQUARE_SIZE_CM, dist=None):
    """
    Recover the camera pose (R, t) in the board's world frame.

    Args:
        scene_img (PIL.Image): image containing a visible checkerboard.
        K (np.ndarray): 3x3 intrinsic matrix from calibrate_intrinsics.
        pattern_size (tuple): (cols, rows) of interior checkerboard corners.
        square_size_cm (float): side length of one checkerboard square.
        dist (np.ndarray): distortion coefficients, default zero.

    Returns:
        (R, t) where R is a 3x3 rotation matrix and t is a 3-vector (world cm).
        Also returns reprojection RMS in pixels as a third value for sanity.
    """
    gray = np.array(scene_img.convert("L"), dtype=np.uint8)
    corners, _ = _find_checkerboard_corners(gray, pattern_size)
    corners = corners.astype(np.float32)

    object_points = _build_object_points(pattern_size, square_size_cm).astype(np.float32)
    dist = np.zeros((1, 5)) if dist is None else np.asarray(dist)

    ok, rvec, tvec = cv2.solvePnP(object_points, corners, K, dist)
    if not ok:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()

    # Reprojection sanity check
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reproj_rms = float(np.sqrt(np.mean(
        np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1))))

    return R, t, reproj_rms


if __name__ == "__main__":

    from pathlib import Path
    from PIL import Image
    from intrinsics import calibrate_intrinsics

    INTRINSIC_DIR = Path("../test-images/intrinsic_calibration")
    SCENE_CALIB   = Path("../test-images/scene6/calibration")

    # Step 1: intrinsic calibration
    intrinsic_imgs = [Image.open(p) for p in sorted(INTRINSIC_DIR.glob("*.png"))]
    K = calibrate_intrinsics(intrinsic_imgs, method="zhang")
    print(f"K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
          f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # Step 2: extrinsic calibration
    scene_calib_path = sorted(SCENE_CALIB.glob("*.png"))[0]
    scene_img = Image.open(scene_calib_path)
    R, t, rms = solve_scene_pose(scene_img, K)

    print(f"\nScene pose from {scene_calib_path.name}:")
    print(f"  R =\n{np.array2string(R, precision=3)}")
    print(f"  t = ({t[0]:+.1f}, {t[1]:+.1f}, {t[2]:+.1f}) cm")
    print(f"  reprojection RMS: {rms:.3f} px")

    cam_world = -R.T @ t
    print(f"  camera position in world: "
          f"({cam_world[0]:+.1f}, {cam_world[1]:+.1f}, {cam_world[2]:+.1f}) cm")

    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert rms < 10.0, f"Reprojection RMS {rms:.2f} px is suspiciously high"
    print("\nExtrinsic calibration OK.")
