import numpy as np

from camera_calibration_zhang import calibrate as _calibrate_zhang
from camera_calibration_dlt import calibrate as _calibrate_dlt


PATTERN_SIZE = (8, 6)
SQUARE_SIZE_CM = 4.0


def calibrate_intrinsics(imgs, method="zhang", **kwargs):
    """
    Recover the 3x3 intrinsic matrix K.

    Args:
        imgs (list<PIL.Image>): calibration images.
            - "zhang": checkerboard images at diverse board orientations.
            - "dlt":   one image with known 2D/3D correspondences (passed via
                       kwargs as points_2d / points_3d).
        method (str): "zhang" or "dlt".
        **kwargs:
            - method="dlt": points_2d (Nx2), points_3d (Nx3) must be supplied.

    Returns:
        np.ndarray: 3x3 intrinsic matrix K.
    """
    if method == "zhang":
        calibration = _calibrate_zhang(imgs, PATTERN_SIZE, SQUARE_SIZE_CM)
        return calibration["K"]

    if method == "dlt":
        if "points_2d" not in kwargs or "points_3d" not in kwargs:
            raise ValueError(
                "method='dlt' requires points_2d and points_3d kwargs")
        calibration = _calibrate_dlt(
            imgs[0],
            points_2d=np.asarray(kwargs["points_2d"], dtype=np.float64),
            points_3d=np.asarray(kwargs["points_3d"], dtype=np.float64),
        )
        return calibration["K"]

    raise ValueError(f"Unknown method: {method!r} (expected 'zhang' or 'dlt')")


if __name__ == "__main__":

    from pathlib import Path
    from PIL import Image

    INTRINSIC_DIR = Path("../test-images/intrinsic_calibration")

    paths = sorted(INTRINSIC_DIR.glob("*.png"))
    imgs = [Image.open(p) for p in paths]
    print(f"Loaded {len(imgs)} intrinsic calibration image(s) from {INTRINSIC_DIR}")

    K = calibrate_intrinsics(imgs, method="zhang")
    print(f"\nK =\n{np.array2string(K, precision=2)}")
    print(f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
          f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    assert K.shape == (3, 3)
    assert K[0, 0] > 0 and K[1, 1] > 0
    print("\nIntrinsic calibration OK.")
