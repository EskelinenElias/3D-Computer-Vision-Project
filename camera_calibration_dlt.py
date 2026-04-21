import json

import cv2
import numpy as np
from scipy.linalg import rq


def select_points(image_rgb, n_points, scale_factor=0.4):
    """
    Interactively pick n_points image-plane correspondences via an OpenCV window.

    Left-click marks a point. ESC or closing the window stops early.
    Returns an (n_points, 2) array of pixel coordinates in the *full-resolution*
    frame (the window is downscaled for display; picks are unscaled on return).
    """
    height, width = image_rgb.shape[:2]
    display_size = (int(width * scale_factor), int(height * scale_factor))
    display = cv2.resize(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        display_size, interpolation=cv2.INTER_AREA,
    ).astype(np.uint8)

    picks = []

    def on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picks.append([x / scale_factor, y / scale_factor])
            cv2.circle(display, (x, y), 5, (255, 255, 255), -1)
            cv2.putText(display, str(len(picks)), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Select Points", display)
            print(f"Point {len(picks)}/{n_points}: ({picks[-1][0]:.1f}, {picks[-1][1]:.1f})")

    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", on_click)
    print(f"Click {n_points} points. ESC to abort.")

    while True:
        cv2.imshow("Select Points", display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or len(picks) >= n_points:
            break
        if cv2.getWindowProperty("Select Points", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    if len(picks) < n_points:
        raise RuntimeError(f"Got {len(picks)} points, need {n_points}")
    return np.asarray(picks[:n_points], dtype=np.float64)


def build_object_points_from_centers(centers, cube_half_size_cm=2.0):
    """
    Generate DLT object points from cube centers of mass (port of the exercise
    12 approach).

    For each center (x, y, z), emits two corners:
      - top-front-left:    (x + half, y - half, z + half)
      - bottom-front-left: (x + half, y - half, z - half)

    Pairs at different Z guarantee a non-coplanar set so DLT is well-conditioned.

    Args:
        centers           : Nx3 array of cube centers of mass in world cm.
        cube_half_size_cm : half of the cube edge length.

    Returns (2N, 3) array of 3D corner points.
    """
    centers = np.asarray(centers, dtype=np.float64)
    h = cube_half_size_cm
    points = []
    for x, y, z in centers:
        points.append([x + h, y - h, z + h])  # Top-Front-Left
        points.append([x + h, y - h, z - h])  # Bottom-Front-Left
    return np.asarray(points)


def _normalization_matrix(points):
    """
    Hartley normalization matrix: translate centroid to origin, scale mean
    distance from origin to sqrt(dim). Works for 2D or 3D point sets.
    """
    dim = points.shape[1]
    centroid = points.mean(axis=0)
    mean_distance = np.mean(np.linalg.norm(points - centroid, axis=1))
    scale = np.sqrt(dim) / (mean_distance + 1e-12)

    T = np.eye(dim + 1) * scale
    T[:dim, dim] = -scale * centroid
    T[dim, dim] = 1.0
    return T


def _solve_dlt(points_2d, points_3d):
    """
    Solve Am = 0 by SVD for the 3x4 projection matrix M mapping 3D→2D.

    Each 2D↔3D pair contributes two rows of A. Needs ≥6 non-coplanar points.
    """
    rows = []
    for (x, y), (X, Y, Z) in zip(points_2d, points_3d):
        rows.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        rows.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
    A = np.asarray(rows, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)


def _dlt_normalized(points_2d, points_3d):
    """DLT with Hartley normalization for numerical conditioning."""
    T2 = _normalization_matrix(points_2d)
    T3 = _normalization_matrix(points_3d)

    pts2_h = np.column_stack([points_2d, np.ones(len(points_2d))])
    pts3_h = np.column_stack([points_3d, np.ones(len(points_3d))])

    pts2_norm = (T2 @ pts2_h.T).T[:, :2]
    pts3_norm = (T3 @ pts3_h.T).T[:, :3]

    M_norm = _solve_dlt(pts2_norm, pts3_norm)
    return np.linalg.inv(T2) @ M_norm @ T3


def _decompose_projection(M):
    """
    Factor a 3x4 projection matrix M into K (3x3 upper-triangular intrinsics),
    R (3x3 rotation), C (3-vector camera center in world).

    M = K [R | -R C], so A = M[:, :3] = K R and b = M[:, 3] = -K R C.
    RQ-decompose A = K R, then C = -A⁻¹ b.
    """
    A, b = M[:, :3], M[:, 3]

    K, R = rq(A)

    # Ensure K has positive diagonal (flip sign of column k and row k together)
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = np.linalg.inv(D) @ R

    # Normalize K so K[2, 2] == 1, propagate sign to R so M = K R is preserved
    sign = np.sign(K[-1, -1])
    K = K * sign
    R = R * sign
    K = K / K[2, 2]

    C = -np.linalg.inv(A) @ b

    return K, R, C


def _project(M, points_3d):
    """Project Nx3 points through the 3x4 matrix M to Nx2 pixel coords."""
    pts_h = np.column_stack([points_3d, np.ones(len(points_3d))])
    proj = (M @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def calibrate(pil_image, points_2d=None, points_3d=None,
              scale_factor=0.4, verbose=True):
    """
    DLT-based camera calibration from a single image.

    Args:
        pil_image   : PIL.Image used both for the manual picker and to get
                      the image size.
        points_2d   : (N, 2) pixel coordinates. If None, opens an interactive
                      picker — `points_3d` must then be provided to determine N.
        points_3d   : (N, 3) world coordinates (cm). Required.
        scale_factor: display-window scale for the interactive picker.

    Returns a dict with:
        K          : (3, 3) intrinsic matrix
        dist       : (1, 5) zeros (DLT does not model distortion)
        image_size : (width, height)
        rvecs      : [rvec] — single-view Rodrigues rotation vector
        tvecs     : [tvec] — single-view translation (3x1)
        M          : (3, 4) full projection matrix (extra vs Zhang's output)
    """
    if points_3d is None:
        raise ValueError("points_3d must be provided")
    points_3d = np.asarray(points_3d, dtype=np.float64)
    if len(points_3d) < 6:
        raise ValueError(f"Need ≥6 correspondences, got {len(points_3d)}")

    image_rgb = np.asarray(pil_image.convert("RGB"))
    height, width = image_rgb.shape[:2]
    image_size = (width, height)

    if points_2d is None:
        points_2d = select_points(image_rgb, n_points=len(points_3d),
                                  scale_factor=scale_factor)
    else:
        points_2d = np.asarray(points_2d, dtype=np.float64)

    if len(points_2d) != len(points_3d):
        raise ValueError(
            f"2D/3D count mismatch: {len(points_2d)} vs {len(points_3d)}")

    M = _dlt_normalized(points_2d, points_3d)
    K, R, C = _decompose_projection(M)

    tvec = -R @ C
    rvec, _ = cv2.Rodrigues(R)

    if verbose:
        reproj = _project(M, points_3d)
        rms = np.sqrt(np.mean(np.sum((reproj - points_2d) ** 2, axis=1)))
        print(f"[dlt] RMS reprojection error: {rms:.3f} px ({len(points_3d)} points)")
        print(f"[dlt] K:\n{np.array2string(K, precision=1)}")
        print(f"[dlt] Camera center C (world): "
              f"({C[0]:+.1f}, {C[1]:+.1f}, {C[2]:+.1f}) cm")

    return {
        "K": K,
        "dist": np.zeros((1, 5)),
        "image_size": image_size,
        "rvecs": [rvec.reshape(3, 1)],
        "tvecs": [tvec.reshape(3, 1)],
        "M": M,
    }


if __name__ == "__main__":

    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt

    FIGS_DIR    = Path("figures")
    SCENE_DIR   = Path("test-images/set4/images")
    POINTS_FILE = Path("test-images/set4/dlt_points.json")
    SQUARE_SIZE_CM = 4.0

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    scene_paths = sorted(SCENE_DIR.glob("*.png"))
    pil_image = Image.open(scene_paths[0])
    print(f"Loaded {scene_paths[0].name} ({pil_image.size[0]}x{pil_image.size[1]})")

    # Default world points: 4 floor corners + 4 cube-top corners (non-coplanar,
    # so DLT is well-conditioned). Adjust to match your actual calibration targets.
    default_points_3d = np.array([
        [0.0,   0.0,   0.0],
        [28.0,  0.0,   0.0],
        [0.0,   -20.0, 0.0],
        [28.0,  -20.0, 0.0],
        [4.0,   -4.0,  4.0],
        [12.0,  -4.0,  4.0],
        [20.0,  -12.0, 4.0],
        [4.0,   -16.0, 4.0],
    ])

    if POINTS_FILE.exists():
        with POINTS_FILE.open() as f:
            saved = json.load(f)
        points_2d = np.asarray(saved["points_2d"], dtype=np.float64)
        points_3d = np.asarray(saved["points_3d"], dtype=np.float64)
        print(f"Loaded {len(points_2d)} saved correspondences from {POINTS_FILE}")
    else:
        print("No saved correspondences found — launching interactive picker.")
        print("For each 3D point below, click its pixel location in the image:")
        for i, (x, y, z) in enumerate(default_points_3d):
            print(f"  {i+1}. world = ({x:+.1f}, {y:+.1f}, {z:+.1f}) cm")
        points_3d = default_points_3d
        points_2d = select_points(
            np.asarray(pil_image.convert("RGB")),
            n_points=len(points_3d), scale_factor=0.4)
        POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with POINTS_FILE.open("w") as f:
            json.dump({"points_2d": points_2d.tolist(),
                       "points_3d": points_3d.tolist()}, f, indent=2)
        print(f"Saved correspondences to {POINTS_FILE}")

    calibration = calibrate(pil_image, points_2d=points_2d, points_3d=points_3d)

    K = calibration["K"]
    w, h = calibration["image_size"]
    assert K.shape == (3, 3), "K must be 3x3"
    assert K[0, 0] > 0 and K[1, 1] > 0, "Focal lengths must be positive"
    assert 0 < K[0, 2] < w and 0 < K[1, 2] < h, "Principal point must lie inside image"
    print("\nAll sanity checks passed.")

    # Visualize: reprojected correspondences + world frame on the scene image
    M = calibration["M"]
    reproj_2d = _project(M, points_3d)

    axis_length_cm = 3 * SQUARE_SIZE_CM
    world_axes = np.array([
        [0, 0, 0],
        [axis_length_cm,  0,              0],
        [0,              -axis_length_cm, 0],
        [0,               0,              axis_length_cm],
    ], dtype=np.float64)
    axes_px = _project(M, world_axes)
    origin_px = axes_px[0]

    plt.figure()
    plt.imshow(pil_image)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=60, facecolors="none",
                edgecolors="lime", linewidths=1.5, label="picked")
    plt.scatter(reproj_2d[:, 0], reproj_2d[:, 1], s=20, c="red",
                marker="x", linewidths=2.0, label="reprojected")
    for axis_index, color, label in [(1, "red", "X"), (2, "green", "Y"), (3, "blue", "Z")]:
        end_px = axes_px[axis_index]
        plt.plot([origin_px[0], end_px[0]], [origin_px[1], end_px[1]],
                 color=color, linewidth=2.5)
        plt.text(end_px[0], end_px[1], label, color=color,
                 fontsize=12, fontweight="bold")
    plt.title("DLT calibration — picked vs reprojected points + world frame")
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "dlt-calibration-world-frame.png",
                dpi=300, bbox_inches="tight")
    plt.show()
