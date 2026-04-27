import numpy as np

GRABBER_OFFSET_CM = 12.0
MIN_TURN_DEG = 0.5
MIN_DRIVE_CM = 0.5


def _normalize_angle(angle_rad):
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def _step(cmd, param, robot_pos, robot_heading):
    """Build a plan step dict with the pose the robot holds *after* the step."""
    return {
        "cmd": cmd,
        "param": param,
        "pos": np.asarray(robot_pos, dtype=float).copy(),
        "heading": float(robot_heading),
    }


def _plan_drive_grabber_to(robot_pos, robot_heading, target_xy):
    """
    Plan the steps to drive the robot so its grabber tip reaches target_xy.

    The robot center ends up GRABBER_OFFSET_CM behind the target along the
    approach direction.

    Returns (steps, new_robot_pos, new_robot_heading).
    """
    steps = []
    target_xy = np.asarray(target_xy, dtype=float)

    delta = target_xy - robot_pos
    distance_to_target = np.linalg.norm(delta)
    if distance_to_target < MIN_DRIVE_CM:
        return steps, robot_pos, robot_heading

    # Turn to face the target
    desired_heading = np.arctan2(delta[1], delta[0])
    turn_deg = np.degrees(_normalize_angle(desired_heading - robot_heading))
    if abs(turn_deg) > MIN_TURN_DEG:
        robot_heading = desired_heading
        steps.append(_step("turn", turn_deg, robot_pos, robot_heading))

    # Drive the grabber to target
    drive_distance = distance_to_target - GRABBER_OFFSET_CM
    if abs(drive_distance) > MIN_DRIVE_CM:
        robot_pos = robot_pos + drive_distance * np.array(
            [np.cos(robot_heading), np.sin(robot_heading)])
        steps.append(_step("go", drive_distance, robot_pos, robot_heading))

    return steps, robot_pos, robot_heading


def _plan_pick_and_place(robot_pos, robot_heading, cube_pos, target_pos):
    """
    Plan the steps to pick up a cube and place it on a target.

    Args:
        robot_pos     : (2,) or (3,) robot center in world cm
        robot_heading : radians
        cube_pos      : (2,) or (3,) cube position in world cm
        target_pos    : (2,) or (3,) target position in world cm

    Returns (steps, new_robot_pos, new_robot_heading) where `steps` is a list
    of plan-step dicts (see `_step`).
    """
    robot_pos = np.asarray(robot_pos[:2], dtype=float)
    robot_heading = float(robot_heading)

    # Drive grabber to cube, grab
    steps, robot_pos, robot_heading = _plan_drive_grabber_to(
        robot_pos, robot_heading, cube_pos[:2])
    steps.append(_step("grab", None, robot_pos, robot_heading))

    # Drive grabber to target, release
    drive_steps, robot_pos, robot_heading = _plan_drive_grabber_to(
        robot_pos, robot_heading, target_pos[:2])
    steps.extend(drive_steps)
    steps.append(_step("let_go", None, robot_pos, robot_heading))

    return steps, robot_pos, robot_heading


def _plan_for_color(color, robot_state, cubes, targets):
    """
    Plan the pick-and-place steps for a single cube colour.

    robot_state is mutated in-place: its "pos" and "heading" are updated to the
    robot's pose after the plan executes, so consecutive calls chain naturally.

    Raises KeyError if the cube or target for `color` isn't in the detections.

    Returns the list of plan-step dicts (see `_step`).
    """
    if color not in cubes:
        raise KeyError(f"No cube detected for colour {color!r}")
    if color not in targets:
        raise KeyError(f"No target detected for colour {color!r}")

    steps, new_pos, new_heading = _plan_pick_and_place(
        robot_state["pos"], robot_state["heading"],
        cubes[color], targets[color])

    robot_state["pos"] = new_pos
    robot_state["heading"] = new_heading
    return steps


def _translate(steps, drive_scale=1.0, drive_bias=0.0,
               turn_scale=1.0, turn_bias=0.0):
    """Convert a list of plan-step dicts into the final command string.

    drive_scale / drive_bias: calibrate `go(d)` distances as
        d_out = drive_scale * d + sign(d) * drive_bias
    turn_scale / turn_bias: calibrate `turn(theta)` angles as
        theta_out = turn_scale * theta + sign(theta) * turn_bias

    Bias is sign-aware so it represents constant under/overshoot regardless of
    direction. To flip turn direction (robot turns the wrong way), pass
    turn_scale=-1.0.
    """
    parts = []
    for step in steps:
        cmd, param = step["cmd"], step["param"]
        if param is None:
            parts.append(f"{cmd}()")
            continue
        if cmd == "go":
            value = drive_scale * param + np.sign(param) * drive_bias
        elif cmd == "turn":
            value = turn_scale * param + np.sign(param) * turn_bias
        else:
            value = param
        parts.append(f"{cmd}({value:.1f})")
    return "; ".join(parts)


if __name__ == "__main__":

    from pathlib import Path
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    from camera_calibration_zhang import compute_intrinsics, compute_extrinsics
    from object_detection import _detect_cubes, _detect_targets, _detect_robot

    INTRINSIC_DIR = Path("test-images/intrinsic_calibration")
    SCENE_DIR     = Path("test-images/scene6")
    BLOCK_ORDER   = ["red", "green", "blue"]

    intrinsic_images = [Image.open(p) for p in sorted(INTRINSIC_DIR.glob("*.png"))]
    K = compute_intrinsics(intrinsic_images)

    extrinsic_path = sorted((SCENE_DIR / "calibration").glob("*.png"))[0]
    R, t, rms = compute_extrinsics(Image.open(extrinsic_path), K)
    print(f"Scene pose from {extrinsic_path.name} — reprojection RMS: {rms:.3f} px")

    scene_paths = sorted((SCENE_DIR / "images").glob("*.png"))
    scene_image = Image.open(scene_paths[0])
    bgr = cv2.cvtColor(np.array(scene_image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    cubes   = _detect_cubes(hsv, K, R, t)
    targets = _detect_targets(hsv, K, R, t)
    robot   = _detect_robot(hsv, K, R, t)

    robot_state = {
        "pos": robot["pos"][:2].copy(),
        "heading": robot["heading"],
    }
    all_steps = []

    print(f"\n=== Planning on {scene_paths[0].name} ===")
    print(f"Robot start: ({robot_state['pos'][0]:+7.1f}, "
          f"{robot_state['pos'][1]:+7.1f}) cm, "
          f"heading: {np.degrees(robot_state['heading']):+6.1f}°")

    for color in BLOCK_ORDER:
        try:
            steps = _plan_for_color(color, robot_state, cubes, targets)
        except KeyError as e:
            print(f"Skipping {color}: {e}")
            continue
        print(f"\n-- {color} --")
        print(f"  cube:   ({cubes[color][0]:+7.1f}, {cubes[color][1]:+7.1f})")
        print(f"  target: ({targets[color][0]:+7.1f}, {targets[color][1]:+7.1f})")
        print(f"  commands: {_translate(steps)}")
        all_steps.extend(steps)

    assert all(step["cmd"] in ("turn", "go", "grab", "let_go")
               for step in all_steps), "Unknown command emitted"
    n_grabs    = sum(1 for step in all_steps if step["cmd"] == "grab")
    n_releases = sum(1 for step in all_steps if step["cmd"] == "let_go")
    assert n_grabs == n_releases, "Each grab must be paired with a let_go"
    print(f"\nEmitted {len(all_steps)} step(s) "
          f"({n_grabs} pick-and-place cycles). All sanity checks passed.")
    print(f"\nFinal command string:\n{_translate(all_steps)}")

    moved_colors = [c for c in BLOCK_ORDER if c in cubes and c in targets]

    def _simulate(steps, start_pos, start_heading, cycle_colors,
                  samples_per_turn_deg=2.0, samples_per_go=8):
        # Dense sampling so turns render as arcs and the held cube travels with the grabber
        pos = np.asarray(start_pos, dtype=float).copy()
        heading = float(start_heading)
        cycle = 0
        held = None
        frames = [(pos.copy(), heading, held)]
        for step in steps:
            if step["cmd"] == "turn":
                total_deg = step["param"]
                n = max(2, int(abs(total_deg) / samples_per_turn_deg) + 1)
                for frac in np.linspace(0.0, 1.0, n)[1:]:
                    h = heading + np.radians(total_deg) * frac
                    frames.append((pos.copy(), h, held))
                heading = step["heading"]
            elif step["cmd"] == "go":
                end_pos = step["pos"]
                for frac in np.linspace(0.0, 1.0, samples_per_go)[1:]:
                    frames.append((pos + frac * (end_pos - pos), heading, held))
                pos = end_pos.copy()
            elif step["cmd"] == "grab":
                held = cycle_colors[cycle] if cycle < len(cycle_colors) else None
                frames.append((pos.copy(), heading, held))
            elif step["cmd"] == "let_go":
                held = None
                frames.append((pos.copy(), heading, held))
                cycle += 1
        return frames

    start_pos     = np.asarray(robot['pos'][:2], dtype=float)
    start_heading = float(robot['heading'])
    frames = _simulate(all_steps, start_pos, start_heading, moved_colors)
    print(f"Generated {len(frames)} animation frames.")

    import matplotlib.animation as animation

    fig_top, ax_top = plt.subplots(figsize=(12, 6))

    for color, xyz in cubes.items():
        ax_top.scatter(xyz[0], xyz[1], s=140, c=color, marker='s', alpha=0.35,
                       edgecolors='none', label=f'cube start:{color}')
    for color, xyz in targets.items():
        ax_top.scatter(xyz[0], xyz[1], s=140, facecolors='none', edgecolors=color,
                       linewidths=5.0, marker='o', label=f'target:{color}')

    center_trail,  = ax_top.plot([], [], color='magenta', linestyle='--',
                                 linewidth=1.2, zorder=1, label='center trail')
    grabber_trail, = ax_top.plot([], [], color='yellow', linewidth=1.5,
                                 zorder=2, label='grabber trail')
    body_line,     = ax_top.plot([], [], color='yellow', linewidth=2.5, zorder=3)
    center_dot     = ax_top.scatter([], [], s=100, c='magenta', marker='o',
                                    edgecolors='none', zorder=4)
    grabber_dot    = ax_top.scatter([], [], s=80, c='yellow', marker='o',
                                    edgecolors='none', zorder=4)
    held_dot       = ax_top.scatter([], [], s=160, marker='s',
                                    edgecolors='none', zorder=5)
    status_text    = ax_top.text(0.02, 0.98, '', transform=ax_top.transAxes,
                                 va='top', ha='left', fontsize=10,
                                 bbox=dict(facecolor='white', alpha=0.75,
                                           edgecolor='gray'))

    all_points = np.vstack([np.array([p for (p, _, _) in frames]),
                            np.array([p + GRABBER_OFFSET_CM
                                      * np.array([np.cos(h), np.sin(h)])
                                      for (p, h, _) in frames])])
    pad = 5.0
    ax_top.set_xlim(all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ax_top.set_ylim(all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)
    ax_top.set_xlabel('X (cm)')
    ax_top.set_ylabel('Y (cm)')
    ax_top.set_title('Robot pick-and-place animation (top-down)')
    ax_top.set_aspect('equal')
    ax_top.grid(True, linestyle=':', alpha=0.5)

    def _animate_top(i):
        past = frames[: i + 1]
        centers = np.array([p for (p, _, _) in past])
        grabbers = np.array([p + GRABBER_OFFSET_CM * np.array([np.cos(h), np.sin(h)])
                             for (p, h, _) in past])
        pos, heading, held = frames[i]
        tip = pos + GRABBER_OFFSET_CM * np.array([np.cos(heading), np.sin(heading)])

        center_trail.set_data(centers[:, 0], centers[:, 1])
        grabber_trail.set_data(grabbers[:, 0], grabbers[:, 1])
        body_line.set_data([pos[0], tip[0]], [pos[1], tip[1]])
        center_dot.set_offsets([pos])
        grabber_dot.set_offsets([tip])
        if held is not None:
            held_dot.set_offsets([tip])
            held_dot.set_facecolor(held)
            held_dot.set_visible(True)
        else:
            held_dot.set_visible(False)
        status_text.set_text(
            f'frame {i+1}/{len(frames)}\n'
            f'pos: ({pos[0]:+.1f}, {pos[1]:+.1f}) cm\n'
            f'heading: {np.degrees(heading):+.1f}°\n'
            f'holding: {held or "—"}')
        return (center_trail, grabber_trail, body_line, center_dot,
                grabber_dot, held_dot, status_text)

    anim_top = animation.FuncAnimation(
        fig_top, _animate_top, frames=len(frames),
        interval=30, blit=False, repeat=False)
    fig_top.tight_layout()
    plt.show()
