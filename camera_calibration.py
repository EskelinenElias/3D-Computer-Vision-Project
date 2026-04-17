import numpy as np
import cv2
from scipy.optimize import least_squares

from checkerboard_detection import _find_checkerboard_corners


def _build_object_points(pattern_size, square_size_cm):
    """Build 3D world coordinates for checkerboard corners on the Z=0 plane.

    Y is flipped so +Z points out of the board toward the camera (OpenCV convention).
    """
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    points[:, 0] = grid[:, 0]
    points[:, 1] = -grid[:, 1]
    points *= square_size_cm
    return points


def _normalize_points(points):
    """Hartley normalization: centroid to origin, mean distance = sqrt(2)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_distance = np.mean(np.linalg.norm(centered, axis=1))
    scale = np.sqrt(2) / (mean_distance + 1e-12)
    transform = np.array([[scale, 0,     -scale * centroid[0]],
                          [0,     scale, -scale * centroid[1]],
                          [0,     0,      1]], dtype=np.float64)
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    normalized = (transform @ points_homogeneous.T).T
    return normalized[:, :2], transform


def _compute_homography_dlt(world_points, image_points):
    """
    Planar DLT homography with Hartley normalization.

    World points are on Z=0, so we solve a 3x3 H mapping (X, Y) -> (u, v).
    """
    assert len(world_points) >= 4
    world_points = world_points.astype(np.float64)
    image_points = image_points.astype(np.float64)

    world_normalized, T_world = _normalize_points(world_points)
    image_normalized, T_image = _normalize_points(image_points)

    constraints = []
    for (X, Y), (u, v) in zip(world_normalized, image_normalized):
        constraints.append([-X, -Y, -1,  0,  0,  0, u*X, u*Y, u])
        constraints.append([ 0,  0,  0, -X, -Y, -1, v*X, v*Y, v])
    constraints = np.array(constraints)

    _, _, Vt = np.linalg.svd(constraints)
    H_normalized = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_image) @ H_normalized @ T_world
    return H / H[2, 2]


def _vij(H, i, j):
    """Zhang's constraint row v_ij (shape (6,)) derived from columns i, j of H."""
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j],
    ], dtype=np.float64)


def _zhang_extract_intrinsics(homographies):
    """
    Closed-form K from a list of planar homographies (Z. Zhang, 2000).

    Each H contributes two constraints on the image of the absolute conic B,
    then K is recovered by Cholesky-like decomposition.
    """
    n_views = len(homographies)
    V = np.zeros((2 * n_views, 6), dtype=np.float64)
    for k, H in enumerate(homographies):
        V[2 * k]     = _vij(H, 0, 1)
        V[2 * k + 1] = _vij(H, 0, 0) - _vij(H, 1, 1)

    _, _, Vt = np.linalg.svd(V)
    B11, B12, B22, B13, B23, B33 = Vt[-1]

    cy    = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lam   = B33 - (B13**2 + cy * (B12 * B13 - B11 * B23)) / B11
    fx    = np.sqrt(abs(lam / B11))
    fy    = np.sqrt(abs(lam * B11 / (B11 * B22 - B12**2)))
    skew  = -B12 * fx**2 * fy / lam
    cx    = skew * cy / fy - B13 * fx**2 / lam

    return np.array([[fx,  skew, cx],
                     [0,   fy,   cy],
                     [0,   0,    1]], dtype=np.float64)


def _extract_extrinsics(K, H):
    """Recover (rvec, tvec) for one view from intrinsic matrix K and homography H."""
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    scale = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = scale * (K_inv @ h1)
    r2 = scale * (K_inv @ h2)
    translation = scale * (K_inv @ h3)

    # Flip if board ended up behind the camera
    if translation[2] < 0:
        r1, r2, translation = -r1, -r2, -translation

    r3 = np.cross(r1, r2)

    # Project the approximate rotation onto SO(3) via SVD
    rotation_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(rotation_approx)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        Vt[-1] *= -1
        rotation = U @ Vt

    rvec, _ = cv2.Rodrigues(rotation)
    return rvec, translation.reshape(3, 1)


def _project_points(object_points, K, rvec, tvec):
    """Project 3D points to 2D pixel coordinates. Returns (N, 2)."""
    rotation, _ = cv2.Rodrigues(rvec)
    points_camera = rotation @ object_points.T + tvec.reshape(3, 1)
    points_image = K @ points_camera
    points_image = points_image[:2] / points_image[2:3]
    return points_image.T


def _K_from_exif(pil_image, image_size):
    """Compute K from EXIF focal length + focal-plane pixel density. Returns K or None."""
    try:
        exif = pil_image._getexif()
        if exif is None:
            return None
        focal_mm = float(exif.get(37386, 0))
        fpx_res = float(exif.get(41486, 0))
        fpy_res = float(exif.get(41487, 0))
        fp_unit = int(exif.get(41488, 2))

        if focal_mm <= 0 or fpx_res <= 0:
            return None

        if fp_unit == 2:  # inches
            fpx_res /= 25.4
            fpy_res /= 25.4
        elif fp_unit == 3:  # cm
            fpx_res /= 10.0
            fpy_res /= 10.0

        w, h = image_size
        fx = focal_mm * fpx_res
        fy = focal_mm * fpy_res
        return np.array([[fx, 0, w / 2],
                         [0, fy, h / 2],
                         [0, 0, 1]], dtype=np.float64)
    except Exception:
        return None


def _default_K(image_size):
    """Fallback K when Zhang's closed-form gives an implausible result."""
    w, h = image_size
    focal = 0.8 * max(w, h)
    return np.array([[focal, 0,     w / 2],
                     [0,     focal, h / 2],
                     [0,     0,     1]], dtype=np.float64)


def _is_plausible_K(K, image_size):
    """Check if K has physically plausible values."""
    w, h = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return (fx > 0 and fy > 0 and
            0 < cx < w and 0 < cy < h and
            fx < 10 * max(w, h) and fy < 10 * max(w, h))


def calibrate(images, pattern_size=(8, 6), square_size_cm=4.0):
    """
    Camera calibration from checkerboard images using Zhang's method.

    Pipeline:
      1. Detect corners in each image.
      2. DLT homography per view.
      3. Zhang's closed-form K.
      4. Per-view extrinsics from K + H.
      5. Joint Levenberg-Marquardt refinement of K + all extrinsics.

    Returns dict with 'K', 'dist', 'image_size', 'rvecs', 'tvecs'.
    """
    object_points_template = _build_object_points(pattern_size, square_size_cm)
    object_points_per_view = []
    image_points_per_view = []
    image_size = None

    for pil_image in images:
        gray = np.array(pil_image.convert("L"), dtype=np.uint8)
        if image_size is None:
            h, w = gray.shape
            image_size = (w, h)
        try:
            corners, _ = _find_checkerboard_corners(gray, pattern_size)
            object_points_per_view.append(object_points_template.copy())
            image_points_per_view.append(corners.astype(np.float32))
        except RuntimeError:
            continue

    if len(object_points_per_view) < 2:
        raise ValueError(
            f"Only {len(object_points_per_view)} views found — need at least 2.")

    n_views = len(object_points_per_view)

    # DLT homographies
    homographies = [
        _compute_homography_dlt(obj_pts[:, :2], img_pts)
        for obj_pts, img_pts in zip(object_points_per_view, image_points_per_view)
    ]

    # Zhang's closed-form K (with 1. EXIF and 2. default fallbacks if the result is implausible)
    K_initial = _zhang_extract_intrinsics(homographies)
    print(f"[calibrate] Zhang's K:\n{np.array2string(K_initial, precision=1)}")
    if not _is_plausible_K(K_initial, image_size):
        K_exif = _K_from_exif(images[0], image_size)
        if K_exif is not None:
            print("[calibrate] Using EXIF-based K estimate")
            K_initial = K_exif
        else:
            print("[calibrate] Fallback to default K estimate")
            K_initial = _default_K(image_size)

    # Initial extrinsics per view
    rvecs_initial, tvecs_initial = [], []
    for H in homographies:
        rvec, tvec = _extract_extrinsics(K_initial, H)
        rvecs_initial.append(rvec)
        tvecs_initial.append(tvec)

    # LM refinement — parameter vector: [fx, fy, cx, cy, rvec_0(3), tvec_0(3), ...]
    w, h = image_size
    params_initial = np.zeros(4 + n_views * 6)
    params_initial[:4] = [K_initial[0, 0], K_initial[1, 1],
                          K_initial[0, 2], K_initial[1, 2]]
    for i in range(n_views):
        base = 4 + i * 6
        params_initial[base    : base + 3] = rvecs_initial[i].ravel()
        params_initial[base + 3: base + 6] = tvecs_initial[i].ravel()

    def residuals(params):
        fx, fy, cx, cy = params[:4]
        K_current = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        all_residuals = []
        for i in range(n_views):
            base = 4 + i * 6
            rvec = params[base    : base + 3].reshape(3, 1)
            tvec = params[base + 3: base + 6].reshape(3, 1)
            projected = _project_points(
                object_points_per_view[i].astype(np.float64), K_current, rvec, tvec)
            observed = image_points_per_view[i].astype(np.float64)
            all_residuals.append((projected - observed).ravel())
        return np.concatenate(all_residuals)

    # Bounds: keep K near initial estimate, principal point within central region
    fx_initial, fy_initial = params_initial[0], params_initial[1]
    lower = np.full_like(params_initial, -np.inf)
    upper = np.full_like(params_initial,  np.inf)
    lower[0], upper[0] = fx_initial * 0.8, fx_initial * 1.2
    lower[1], upper[1] = fy_initial * 0.8, fy_initial * 1.2
    lower[2], upper[2] = w * 0.3, w * 0.7
    lower[3], upper[3] = h * 0.3, h * 0.7

    result = least_squares(residuals, params_initial, bounds=(lower, upper), method='trf')

    fx, fy, cx, cy = result.x[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    rvecs, tvecs = [], []
    for i in range(n_views):
        base = 4 + i * 6
        rvecs.append(result.x[base    : base + 3].reshape(3, 1))
        tvecs.append(result.x[base + 3: base + 6].reshape(3, 1))

    rms_error = np.sqrt(np.mean(result.fun ** 2))
    print(f"[calibrate] RMS reprojection error: {rms_error:.4f} px ({n_views} views)")

    return {
        "K": K,
        "dist": np.zeros((1, 5)),
        "image_size": image_size,
        "rvecs": rvecs,
        "tvecs": tvecs,
    }


if __name__ == "__main__":

    import os, glob
    from PIL import Image
    import matplotlib.pyplot as plt

    # Define constants
    FIGS_DIR = os.path.join("figures")
    CALIBRATION_DIR = 'test-images/set2/calibration'
    PATTERN_SIZE = (8, 6)
    SQUARE_SIZE_CM = 4.0

    # Load calibration images
    calib_paths = sorted(glob.glob(os.path.join(CALIBRATION_DIR, '*.png')))
    calib_images = [Image.open(p) for p in calib_paths]
    print(f"Loaded {len(calib_images)} calibration image(s) from {CALIBRATION_DIR}")

    # Run calibration
    calibration = calibrate(calib_images, PATTERN_SIZE, SQUARE_SIZE_CM)

    # Report results
    K = calibration["K"]
    rvecs = calibration["rvecs"]
    tvecs = calibration["tvecs"]
    w, h = calibration["image_size"]
    print("\n=== Calibration result ===")
    print(f"Image size:    {w} x {h}")
    print(f"Intrinsics K:\n{np.array2string(K, precision=2)}")
    print(f"Focal length:  fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}")
    print(f"Principal pt:  cx={K[0, 2]:.1f}, cy={K[1, 2]:.1f}")
    print(f"Views used:    {len(rvecs)}")

    # Sanity assertions
    assert K.shape == (3, 3), "K must be 3x3"
    assert K[0, 0] > 0 and K[1, 1] > 0, "Focal lengths must be positive"
    assert 0 < K[0, 2] < w and 0 < K[1, 2] < h, "Principal point must lie inside image"
    assert len(rvecs) == len(calib_images), "Expected one rvec per view"
    print("\nAll sanity checks passed.")

    # Visualize reprojection + world frame on each view
    object_points = _build_object_points(PATTERN_SIZE, SQUARE_SIZE_CM)
    axis_length_cm = 3 * SQUARE_SIZE_CM
    world_axes = np.array([[0, 0, 0],
                           [axis_length_cm, 0, 0],
                           [0, axis_length_cm, 0],
                           [0, 0, axis_length_cm]], dtype=np.float64)

    for i, pil_image in enumerate(calib_images):
        gray = np.array(pil_image.convert("L"), dtype=np.uint8)
        try:
            detected_corners, _ = _find_checkerboard_corners(gray, PATTERN_SIZE)
        except RuntimeError:
            continue

        projected_corners = _project_points(object_points.astype(np.float64),
                                            K, rvecs[i], tvecs[i])
        per_view_rms = np.sqrt(np.mean(
            np.sum((projected_corners - detected_corners) ** 2, axis=1)))

        axes_image = _project_points(world_axes, K, rvecs[i], tvecs[i])
        origin_px = axes_image[0]

        plt.figure()
        plt.imshow(pil_image)
        plt.scatter(detected_corners[:, 0], detected_corners[:, 1], s=30, facecolors='none', 
                    edgecolors='green', linewidths=1.2, label='detected')
        plt.scatter(projected_corners[:, 0], projected_corners[:, 1], s=6, 
                    c='red', marker='x', label='reprojected')
        for axis_index, color, label in [(1, 'red', 'X'), (2, 'green', 'Y'), (3, 'blue', 'Z')]:
            end_px = axes_image[axis_index]
            plt.plot([origin_px[0], end_px[0]], [origin_px[1], end_px[1]], color=color, linewidth=2.5)
            plt.text(end_px[0], end_px[1], label, color=color, fontsize=12, fontweight='bold')
        plt.title(f"View {i}  —  per-view RMS: {per_view_rms:.2f} px")
        plt.legend(loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS_DIR, f"camera-calibration-{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.show()


