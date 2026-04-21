import cv2


def _downscale(image_gray, max_dim=1024):
    """Downscale image for faster detection. Returns (downscaled_image, scale)."""
    h, w = image_gray.shape
    if max(h, w) <= max_dim:
        return image_gray, 1.0
    scale = max_dim / max(h, w)
    downscaled_image = cv2.resize(image_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return downscaled_image, scale

def _find_checkerboard_corners(image_gray, pattern_size=(8, 6)):
    """
    Detect checkerboard corners with sub-pixel refinement.

    Tries OpenCV's SB detector first, falls back to the classic detector.
    Raises RuntimeError if the board is not found.

    Returns (corners, found) where corners is (N, 2) float32.
    """
    downscaled_image, scale = _downscale(image_gray)

    found, corners = cv2.findChessboardCornersSB(downscaled_image, pattern_size)
    if not found:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FILTER_QUADS |
                 cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(
            downscaled_image, pattern_size, None, flags)

    if not found:
        raise RuntimeError(
            f"Could not detect checkerboard (pattern {pattern_size})")

    # Map back to full resolution and refine to sub-pixel
    corners = corners / scale
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2), True

if __name__ == "__main__":

    from pathlib import Path
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    FIGS_DIR = Path("../figures")
    CALIBRATION_DIR = Path("../test-images/intrinsic_calibration")
    PATTERN_SIZE = (8, 6)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    calib_paths = sorted(CALIBRATION_DIR.glob("*.png"))
    calib_images = [Image.open(p) for p in calib_paths]
    print(f"Loaded {len(calib_images)} image(s) from {CALIBRATION_DIR}")

    for i, (path, image) in enumerate(zip(calib_paths, calib_images)):
        gray_image = np.array(image.convert("L"), dtype=np.uint8)

        try:
            corners, _ = _find_checkerboard_corners(gray_image, PATTERN_SIZE)
        except RuntimeError:
            print(f"Corners not detected in {path.name}")
            continue

        plt.figure()
        plt.imshow(image)
        plt.scatter(corners[:, 0], corners[:, 1], s=4, c="red", marker=".")
        plt.title(f"{path.name}: {len(corners)} corners")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(FIGS_DIR / f"checkerboard-detection-{i+1}.png",
                    dpi=300, bbox_inches="tight")
        plt.show()
