import cv2
import numpy as np

COLOR_RANGES = {
    "red":    [((0,   100, 60), (10,  255, 255)),
               ((165, 100, 60), (179, 255, 255))], # Red has circular hue in HSV
    "green":  [((35,  30,  25), (90,  255, 255))],
    "blue":   [((95,  80,  40), (135, 255, 255))],
    "yellow": [((20,  100, 80), (35,  255, 255))],
    "purple": [((130, 60,  40), (160, 255, 255))],
}

ROBOT_MARKER_FRONT = "yellow"
ROBOT_MARKER_BACK = "purple"
ROBOT_FRONT_HEIGHT_CM = 8.5
ROBOT_BACK_HEIGHT_CM = 9.0
CUBE_TOP_HEIGHT_CM = 4.0
CUBE_COLORS = ("red", "green", "blue")


def _color_mask(hsv_image, color_name):
    """Get morphologically structured binary mask for a named colour."""
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for low, high in COLOR_RANGES[color_name]:
        mask |= cv2.inRange(hsv_image, np.array(low), np.array(high))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _find_blobs(mask, min_area=500, has_hole=None):
    """
    Find colour blobs in a binary mask using contour hierarchy.

    has_hole: None = any, True = only blobs with a child contour, False = only solid.

    Returns list of contours, filtered and sorted largest-first.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    result = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < min_area:
            continue
        has_parent = hierarchy[0][i][3] != -1
        if has_parent:
            continue
        has_child = hierarchy[0][i][2] != -1
        if has_hole is None or has_child == has_hole:
            result.append(contour)

    result.sort(key=cv2.contourArea, reverse=True)
    return result


def _blob_centroid(contour):
    """Pixel centroid (u, v) of a contour, or None if degenerate."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _pixel_to_plane(u, v, K, R, t, plane_z_world=0.0):
    """
    Back-project pixel (u, v) onto the world plane Z = plane_z_world.

    Returns 3D world point (x, y, plane_z_world).
    """
    ray_camera = np.linalg.inv(K) @ np.array([u, v, 1.0])
    camera_origin_world = -R.T @ t
    ray_world = R.T @ ray_camera

    # Parametric: P = camera_origin + s * ray_world, solve for P[2] == plane_z_world
    s = (plane_z_world - camera_origin_world[2]) / ray_world[2]
    return camera_origin_world + s * ray_world

def _world_to_pixel(world_xyz, K, R, t):
    """Project a 3D world point to (u, v) pixel coordinates via K, R, t."""
    camera_pt = R @ np.asarray(world_xyz, dtype=float) + np.asarray(t, dtype=float).ravel()
    image_pt = K @ camera_pt
    return image_pt[0] / image_pt[2], image_pt[1] / image_pt[2]

def _largest_blob_centroid(hsv, color_name, has_hole=None):
    """Pixel centroid of the largest blob of a given colour, or None."""
    blobs = _find_blobs(_color_mask(hsv, color_name), has_hole=has_hole)
    return _blob_centroid(blobs[0]) if blobs else None


def _two_blob_centroid(hsv, color_name):
    """
    Average pixel centroid of the two largest blobs of a given colour.

    The robot front marker is split into two visible blobs; we need to average their
    centroids to get a stable marker location. Falls back to the single-blob centroid 
    if only one blob is visible.
    """
    blobs = _find_blobs(_color_mask(hsv, color_name))
    if not blobs:
        return None
    if len(blobs) < 2:
        return _blob_centroid(blobs[0])
    c1, c2 = _blob_centroid(blobs[0]), _blob_centroid(blobs[1])
    if c1 is None or c2 is None:
        return c1 or c2
    return ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)


def _detect_cubes(hsv, K, R, t):
    """
    Detect coloured cubes (solid blobs, no inner hole).

    The visible colour blob is the cube's top face, so we back-project the
    centroid onto the plane at the cube-top height (not the floor).
    Returns dict colour -> world point (x, y, z) in cm.
    """
    cubes = {}
    for color in CUBE_COLORS:
        uv = _largest_blob_centroid(hsv, color, has_hole=False)
        if uv is not None:
            cubes[color] = _pixel_to_plane(*uv, K, R, t, plane_z_world=CUBE_TOP_HEIGHT_CM)
    return cubes


def _detect_targets(hsv, K, R, t):
    """
    Detect coloured target rings (blobs with an inner hole).
    Returns dict colour -> world point (x, y, z) in cm.
    """
    targets = {}
    for color in CUBE_COLORS:
        uv = _largest_blob_centroid(hsv, color, has_hole=True)
        if uv is not None:
            targets[color] = _pixel_to_plane(*uv, K, R, t, plane_z_world=0.0)
    return targets


def _detect_robot(hsv, K, R, t):
    """
    Detect robot via front/back coloured markers.

    Returns a dict with:
      pos      - rotation centre in world coordinates (= back/purple marker)
      heading  - heading in radians, from back marker toward front marker
      front    - world position of the front (yellow) marker
      back     - world position of the back (purple) marker
    When only one marker is visible, front and back collapse to the same point
    and heading defaults to 0.
    """
    front_px = _two_blob_centroid(hsv, ROBOT_MARKER_FRONT)
    back_px  = _largest_blob_centroid(hsv, ROBOT_MARKER_BACK)

    if front_px is None and back_px is None:
        raise RuntimeError("Could not detect robot markers.")

    front_w = (_pixel_to_plane(*front_px, K, R, t, plane_z_world=ROBOT_FRONT_HEIGHT_CM)
               if front_px is not None else None)
    back_w  = (_pixel_to_plane(*back_px, K, R, t, plane_z_world=ROBOT_BACK_HEIGHT_CM)
               if back_px is not None else None)

    if front_w is not None and back_w is not None:
        heading = np.arctan2(front_w[1] - back_w[1], front_w[0] - back_w[0])
    else:
        # Only one marker visible — collapse front/back to the same point, heading unknown
        fallback = front_w if front_w is not None else back_w
        front_w = back_w = fallback
        heading = 0.0

    return {"pos": back_w, "heading": heading, "front": front_w, "back": back_w}


if __name__ == "__main__":

    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt
    from camera_calibration_zhang import compute_intrinsics, compute_extrinsics

    FIGS_DIR = Path("figures")
    INTRINSIC_DIR = Path("test-images/intrinsic_calibration")
    SCENE_DIR = Path("test-images/scene6")
    SQUARE_SIZE_CM = 4.0

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Solve camera intrinsic matrix
    intrinsic_images = [Image.open(p) for p in sorted(INTRINSIC_DIR.glob("*.png"))]
    K = compute_intrinsics(intrinsic_images)

    # Solve camera extrinsic matrix
    extrinsic_path = sorted((SCENE_DIR / "calibration").glob("*.png"))[0]
    extrinsic_image = Image.open(extrinsic_path)
    R, t, rms = compute_extrinsics(extrinsic_image, K)
    print(f"Scene pose from {extrinsic_path.name} — reprojection RMS: {rms:.3f} px")

    # Load the task image and detect
    scene_paths = sorted((SCENE_DIR / "images").glob("*.png"))
    scene_image = Image.open(scene_paths[0])
    bgr = cv2.cvtColor(np.array(scene_image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Run detection
    try:
        cubes = _detect_cubes(hsv, K, R, t)
        targets = _detect_targets(hsv, K, R, t)
        robot = _detect_robot(hsv, K, R, t)
    except RuntimeError as e:
        print(f"Robot detection failed: {e}")
        robot = None

    # Report results
    print(f"\nObject detection on {scene_paths[0].name}:\n")
    print(f"Cubes ({len(cubes)} detected):")
    for color, xyz in cubes.items():
        print(f" - {color:6s}: ({xyz[0]:+7.1f}, {xyz[1]:+7.1f}, {xyz[2]:+5.1f}) cm")
    print(f"Targets ({len(targets)} detected):")
    for color, xyz in targets.items():
        print(f" - {color:6s}: ({xyz[0]:+7.1f}, {xyz[1]:+7.1f}, {xyz[2]:+5.1f}) cm")
    if robot is not None:
        print(f"Robot pos: ({robot['pos'][0]:+7.1f}, {robot['pos'][1]:+7.1f}, "
              f"{robot['pos'][2]:+5.1f}) cm, heading: {np.degrees(robot['heading']):+6.1f}°")

    plt.figure()
    plt.imshow(scene_image)

    axis_length_cm = SQUARE_SIZE_CM
    origin_px = _world_to_pixel([0.0, 0.0, 0.0], K, R, t)
    for i, (axis_vec, color, label) in enumerate([([axis_length_cm, 0, 0], 'red',   'X'),
                                                   ([0, axis_length_cm, 0], 'green', 'Y'),
                                                   ([0, 0, axis_length_cm], 'blue',  'Z')]):
        end_px = _world_to_pixel(axis_vec, K, R, t)
        plt.plot([origin_px[0], end_px[0]], [origin_px[1], end_px[1]],
                 color=color, linewidth=2.5, zorder=i+2)
        plt.text(end_px[0], end_px[1], label, color=color,
                 fontsize=12, fontweight='bold', zorder=i+2)

    for color, xyz in cubes.items():
        u, v = _world_to_pixel(xyz, K, R, t)
        plt.scatter(u, v, s=140, facecolors=color, edgecolors=color,
                    linewidths=2.0, marker='s', label=f'cube:{color}')
    for color, xyz in targets.items():
        u, v = _world_to_pixel(xyz, K, R, t)
        plt.scatter(u, v, s=140, facecolors='none', edgecolors=color,
                    linewidths=5.0, marker='o', label=f'target:{color}')
    if robot is not None:
        uf, vf = _world_to_pixel(robot['front'], K, R, t)
        ub, vb = _world_to_pixel(robot['back'], K, R, t)
        plt.scatter(uf, vf, s=120, c='yellow', marker='.', linewidths=2.5, zorder=1, label='robot front')
        plt.scatter(ub, vb, s=120, c='magenta', marker='.', linewidths=2.5, zorder=2, label='robot back')
        plt.plot([ub, uf], [vb, vf], color='yellow', linewidth=3, zorder=1)

    plt.title(f"Detections on {scene_paths[0].name}")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "object-detection.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot([0, axis_length_cm], [0, 0], color='red',   linewidth=2.5)
    plt.plot([0, 0], [0, axis_length_cm], color='green', linewidth=2.5)
    plt.text(axis_length_cm, 0, 'X', color='red',   fontsize=12, fontweight='bold')
    plt.text(0, axis_length_cm, 'Y', color='green', fontsize=12, fontweight='bold')

    for color, xyz in cubes.items():
        plt.scatter(xyz[0], xyz[1], s=140, facecolors=color, edgecolors=color,
                    linewidths=2.0, marker='s', label=f'cube:{color}')
    for color, xyz in targets.items():
        plt.scatter(xyz[0], xyz[1], s=140, facecolors='none', edgecolors=color,
                    linewidths=5.0, marker='o', label=f'target:{color}')
    if robot is not None:
        bx, by, bz = robot['back']
        fx_, fy_, _ = robot['front']
        plt.plot([bx, fx_], [by, fy_], color='yellow', linewidth=3, zorder=1)
        plt.scatter(fx_, fy_, s=120, c='yellow', marker='.', linewidths=2.5,
                    zorder=1, label='robot front')
        plt.scatter(bx, by, s=120, c='magenta', marker='.', linewidths=2.5,
                    zorder=2, label='robot back')

    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Top-down world view')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "top-down-view.png", dpi=300, bbox_inches='tight')
    plt.show()
