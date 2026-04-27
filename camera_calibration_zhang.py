import numpy as np
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


def _compute_planar_homography(world_points, image_points):
    """Planar DLT homography from (X, Y) on Z=0 to (u, v), Hartley-normalized."""
    assert len(world_points) >= 4
    world_points = world_points.astype(np.float64)
    image_points = image_points.astype(np.float64)

    world_normalized, T_world = _normalize_points(world_points)
    image_normalized, T_image = _normalize_points(image_points)

    constraints = []
    for (X, Y), (u, v) in zip(world_normalized, image_normalized):
        constraints.append([ X,  Y,  1,  0,  0,  0, -u*X, -u*Y, -u])
        constraints.append([ 0,  0,  0,  X,  Y,  1, -v*X, -v*Y, -v])
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
    """Closed-form K from a list of planar homographies (Zhang, 2000)."""
    n_views = len(homographies)
    V = np.zeros((2 * n_views, 6), dtype=np.float64)
    for k, H in enumerate(homographies):
        V[2 * k]     = _vij(H, 0, 1)
        V[2 * k + 1] = _vij(H, 0, 0) - _vij(H, 1, 1)

    _, _, Vt = np.linalg.svd(V)
    B11, B12, B22, B13, B23, B33 = Vt[-1]

    cy = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lam = B33 - (B13**2 + cy * (B12 * B13 - B11 * B23)) / B11
    fx = np.sqrt(abs(lam / B11))
    fy = np.sqrt(abs(lam * B11 / (B11 * B22 - B12**2)))
    skew = -B12 * fx**2 * fy / lam
    cx = skew * cy / fy - B13 * fx**2 / lam

    return np.array([[fx, skew, cx], 
                     [ 0,  fy,  cy], 
                     [ 0,   0,   1]], dtype=np.float64)


def _rodrigues(x):
    """Convert between axis-angle vector (3,) and rotation matrix (3,3)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape == (3, 3):
        cos_theta = np.clip((np.trace(x) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if theta < 1e-8:
            return np.zeros(3)
        axis = np.array([x[2, 1] - x[1, 2],
                         x[0, 2] - x[2, 0],
                         x[1, 0] - x[0, 1]]) / (2.0 * np.sin(theta))
        return axis * theta

    rvec = x.ravel()
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.eye(3)
    k = rvec / theta
    K_skew = np.array([[    0, -k[2],  k[1]],
                       [ k[2],     0, -k[0]],
                       [-k[1],  k[0],     0]])
    return np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)


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

    rvec = _rodrigues(rotation)
    return rvec.reshape(3, 1), translation.reshape(3, 1)


def _project_points(object_points, K, rvec, tvec):
    """Project 3D points to 2D pixel coordinates. Returns (N, 2)."""
    rotation = _rodrigues(rvec)
    points_camera = rotation @ object_points.T + tvec.reshape(3, 1)
    points_image = K @ points_camera
    points_image = points_image[:2] / points_image[2:3]
    return points_image.T


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


def compute_intrinsics(images, pattern_size=(8, 6), square_size_cm=4.0):
    """
    Recover the camera intrinsic matrix K via Zhang's method.

    Pipeline:
      1. Detect corners in each image.
      2. Planar DLT homography per view.
      3. Zhang's closed-form K.
      4. Initial per-view extrinsics from K + H.
      5. Joint Levenberg-Marquardt refinement of K + all extrinsics.

    Returns the 3x3 intrinsic matrix K.
    """

    # 1. Build object (world) points, find checkerboard corners (image points) from each image

    object_points_template = _build_object_points(pattern_size, square_size_cm)
    object_points_per_view, image_points_per_view = [], []
    image_size = None

    for image in images:
        gray = np.array(image.convert("L"), dtype=np.uint8)
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
        raise ValueError(f"Only {len(object_points_per_view)} views found - need at least 2.")

    # 2. Compute planar homographies between each set of image points and the world points

    n_views = len(object_points_per_view)

    homographies = [
        _compute_planar_homography(obj_pts[:, :2], img_pts)
        for obj_pts, img_pts in zip(object_points_per_view, image_points_per_view)
    ]

    # 3. Solve K using Zhang's closed form method

    K_initial = _zhang_extract_intrinsics(homographies)
    print(f"[compute_intrinsics] Zhang's K:\n{np.array2string(K_initial, precision=1)}")
    if not _is_plausible_K(K_initial, image_size):
        print("[compute_intrinsics] Fallback to default K estimate")
        K_initial = _default_K(image_size)

    # 4. Initial per-view extrinsics

    rvecs_initial, tvecs_initial = [], []
    for H in homographies:
        rvec, tvec = _extract_extrinsics(K_initial, H)
        rvecs_initial.append(rvec)
        tvecs_initial.append(tvec)

    w, h = image_size
    f_initial = 0.5 * (K_initial[0, 0] + K_initial[1, 1])
    params_initial = np.zeros(3 + n_views * 6)
    params_initial[:3] = [f_initial, K_initial[0, 2], K_initial[1, 2]]
    for i in range(n_views):
        base = 3 + i * 6
        params_initial[base    : base + 3] = rvecs_initial[i].ravel()
        params_initial[base + 3: base + 6] = tvecs_initial[i].ravel()

    # 5. Levenberg-Marquardt refinement - adjust parameters to minimize squared reprojection errors

    def residuals(params):
        f, cx, cy = params[:3]
        K_current = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        all_residuals = []
        for i in range(n_views):
            base = 3 + i * 6
            rvec = params[base    : base + 3].reshape(3, 1)
            tvec = params[base + 3: base + 6].reshape(3, 1)
            projected = _project_points(
                object_points_per_view[i].astype(np.float64), K_current, rvec, tvec)
            observed = image_points_per_view[i].astype(np.float64)
            all_residuals.append((projected - observed).ravel())
        return np.concatenate(all_residuals)

    # Initial quesses
    lower = np.full_like(params_initial, -np.inf)
    upper = np.full_like(params_initial,  np.inf)
    lower[0], upper[0] = f_initial * 0.3, f_initial * 3.0
    lower[1], upper[1] = w * 0.3, w * 0.7
    lower[2], upper[2] = h * 0.3, h * 0.7

    result = least_squares(residuals, params_initial, bounds=(lower, upper), method='trf')

    f, cx, cy = result.x[:3]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    rvecs, tvecs = [], []
    for i in range(n_views):
        base = 3 + i * 6
        rvecs.append(result.x[base    : base + 3].reshape(3, 1))
        tvecs.append(result.x[base + 3: base + 6].reshape(3, 1))

    rms_error = np.sqrt(np.mean(result.fun ** 2))
    print(f"[compute_intrinsics] RMS reprojection error: {rms_error:.4f} px ({n_views} views)")

    return K


def compute_extrinsics(scene_img, K, pattern_size=(8, 6), square_size_cm=4.0):
    """
    Recover the camera pose (R, t) in the board's world frame.

    Pipeline:
      1. Detect checkerboard corners in the scene image.
      2. Planar DLT homography from board (Z=0) to pixel coords.
      3. Initial (R, t) from H + K via `_extract_extrinsics`.
      4. Levenberg-Marquardt refinement.

    Returns (R, t, reproj_rms) in world cm and pixels.
    """

    # 1. Build object (world) points, find checkerboard corners (image points) from each image
    
    gray = np.array(scene_img.convert("L"), dtype=np.uint8)
    corners, _ = _find_checkerboard_corners(gray, pattern_size)
    corners = corners.astype(np.float64)
    object_points = _build_object_points(pattern_size, square_size_cm).astype(np.float64)

    # 2. Compute planar DLT homography

    H = _compute_planar_homography(object_points[:, :2], corners)

    # 3. Extract inintial extrensics

    rvec_init, tvec_init = _extract_extrinsics(K, H)
    params_initial = np.concatenate([rvec_init.ravel(), tvec_init.ravel()])

    # 4. LM refinement

    def residuals(params):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)
        return (_project_points(object_points, K, rvec, tvec) - corners).ravel()

    result = least_squares(residuals, params_initial, method="lm")
    rvec = result.x[:3].reshape(3, 1)
    tvec = result.x[3:].reshape(3, 1)

    R = _rodrigues(rvec)
    t = tvec.ravel()
    reproj_rms = float(np.sqrt(np.mean(result.fun ** 2)))
    return R, t, reproj_rms


if __name__ == "__main__":

    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt

    FIGS_DIR = Path("figures")
    CALIBRATION_DIR = Path("test-images/intrinsics-2")
    SCENE_IMAGE = Path("test-images/intrinsics-2/IMG_1361.JPG")
    PATTERN_SIZE = (8, 6)
    SQUARE_SIZE_CM = 4.0
    AXIS_LENGTH_CM = 3 * SQUARE_SIZE_CM
    FILETYPE = ".JPG"

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    calib_paths = sorted(CALIBRATION_DIR.glob(f"*{FILETYPE}"))
    calib_images = [Image.open(p) for p in calib_paths]
    print(f"Loaded {len(calib_images)} calibration image(s) from {CALIBRATION_DIR}")

    K = compute_intrinsics(calib_images, PATTERN_SIZE, SQUARE_SIZE_CM)
    w, h = calib_images[0].size
    print(f"\nIntrinsics K:\n{np.array2string(K, precision=2)}")
    print(f"Focal length:  fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}")
    print(f"Principal pt:  cx={K[0, 2]:.1f}, cy={K[1, 2]:.1f}")

    assert K.shape == (3, 3), "K must be 3x3"
    assert K[0, 0] > 0 and K[1, 1] > 0, "Focal lengths must be positive"
    assert 0 < K[0, 2] < w and 0 < K[1, 2] < h, "Principal point must lie inside image"
    print("\nIntrinsic calibration OK.")

    overlay_image = Image.open(SCENE_IMAGE) if SCENE_IMAGE.exists() else calib_images[-1]
    R, t, rms = compute_extrinsics(overlay_image, K, PATTERN_SIZE, SQUARE_SIZE_CM)
    print(f"Extrinsic RMS: {rms:.3f} px")

    world_frame = np.array([[0, 0, 0],
                            [AXIS_LENGTH_CM,  0,              0],
                            [0,              AXIS_LENGTH_CM, 0],
                            [0,               0,              AXIS_LENGTH_CM]], dtype=np.float64)
    axes_cam = (R @ world_frame.T + t.reshape(3, 1))
    axes_img = K @ axes_cam
    axes_image = (axes_img[:2] / axes_img[2:3]).T
    origin_px = axes_image[0]

    plt.figure()
    plt.imshow(overlay_image)
    for axis_index, color, label in [(1, "red", "X"), (2, "green", "Y"), (3, "blue", "Z")]:
        end_px = axes_image[axis_index]
        plt.plot([origin_px[0], end_px[0]], [origin_px[1], end_px[1]],
                 color=color, linewidth=2.5)
        plt.text(end_px[0], end_px[1], label, color=color,
                 fontsize=12, fontweight="bold")
    plt.title("Calibrated world frame overlaid on scene image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "camera-calibration-world-frame.png",
                dpi=300, bbox_inches="tight")
    plt.show()

