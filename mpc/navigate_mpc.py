"""
Stage 11 extension: drive the MPC controller (zero RL residual -- pure classical locomotion)
toward a world-frame (x, y) goal, reusing navigate.py's own proportional steering law
(compute_nav_command) unmodified. Mirrors navigate.py's Stage 10a scope exactly (point-to-goal,
optionally across terrain; no obstacle avoidance yet -- MPCResidualEnv has no nav_obstacles
injection mechanism, unlike Go1FlatEnv).

Why this comparison is interesting, not just a port: Stage 10c found RL never learned to sustain
a real turn (see HISTORY.md), so envs/go1_env.py's own navigation always drives via
(target_speed, target_lateral_speed) only -- crabbing/strafing toward the goal rather than
turning to face it. MPC sustains real commanded yaw-rate cleanly (1.955 rad/s, see
verify_residual_interface.py), but this first pass deliberately reuses compute_nav_command()
UNCHANGED (speed+lateral only, no yaw-rate) for an apples-to-apples test of whether MPC's own
more stable low-level tracking helps navigation at all, before trying a turn-to-face steering law
that would actually exercise the capability gap.

Found and fixed a real compatibility gap along the way: compute_nav_command() recomputes the
command EVERY control tick (200Hz here, vs envs/go1_env.py's own 50Hz), continuously decreasing
it as the goal nears. A raw step-commanded change this env's own zero-residual eval harness never
exercises (which always holds ONE fixed command for a whole episode) -- confirmed directly: the
exact same (forward=0.8, lateral=0.3) combo that runs clean for a full fixed 10s episode on 4 of 5
seeds caused a QP infeasibility (not a fall -- the classical controller's receding-horizon
dynamics-model prediction assumes an approximately constant commanded velocity over its own
horizon, and degrades under commands that keep shifting tick to tick) partway through 4 of 8
navigation seeds tested. Fixed with a simple rate limiter (`--max-accel`, default 1.5 m/s^2,
applied to both forward and lateral commands before `set_command`) -- real robot navigation
stacks apply exactly this kind of acceleration limit on a velocity reference for the same reason.
Verified directly: the same 8 seeds that included 4 QP-infeasibility failures all ran clean with
the rate limiter active (final distance 0.025-0.028m, well within tolerance).

`--steering turn` is the turn-to-face follow-up flagged above: rotate to face the goal bearing
(commanded yaw-rate, proportional to heading error) while driving forward at a speed tapered by
how well-aligned the heading already is (`cos(heading_err)`, clipped to >=0 -- never drives
backward, just turns around instead) -- the capability `--steering strafe` (the default, exactly
reproducing the original compute_nav_command()-based behavior) deliberately leaves unused. Same
rate-limiting fix applied to the yaw-rate command too (`--max-yaw-accel`), for the identical reason
(a continuously-recomputed command destabilizes this env's QP -- see above).

Usage:
    python3 navigate_mpc.py --goal-x 3.0 --goal-y 1.0 --record out.gif
    python3 navigate_mpc.py --goal-x 4.0 --goal-y 0.0 --terrain-amplitude 0.08 --record out.gif
    python3 navigate_mpc.py --goal-x 3.0 --goal-y -2.0 --steering turn --record turn.gif
"""
import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mpc_residual_env import MPCResidualEnv
from navigate import compute_nav_command


def compute_nav_command_turn(x, y, yaw, goal_x, goal_y, kp_fwd, kp_yaw, speed_lo, speed_hi,
                              yaw_lo, yaw_hi):
    """Turn-to-face steering: command a yaw-rate proportional to the heading error toward the
    goal bearing, and a forward speed tapered by how well-aligned the heading already is
    (cos(heading_err), floored at 0 -- never drives backward, turns around instead). Unlike
    compute_nav_command's strafe/crab law (reused unmodified for --steering strafe), this
    exercises the MPC base controller's actually-working yaw-rate tracking -- RL's own
    navigation (envs/go1_env.py) has no equivalent, since Stage 10c found RL never learned to
    sustain a commanded turn."""
    dx, dy = goal_x - x, goal_y - y
    dist = float(np.hypot(dx, dy))
    heading_err = float(np.arctan2(dy, dx)) - yaw
    heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
    yaw_cmd = float(np.clip(kp_yaw * heading_err, yaw_lo, yaw_hi))
    fwd_cmd = float(np.clip(kp_fwd * dist, speed_lo, speed_hi) * max(0.0, np.cos(heading_err)))
    return fwd_cmd, yaw_cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--waypoint", type=float, nargs=2, action="append", default=None,
                         metavar=("X", "Y"))
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--speed-range", type=float, nargs=2, default=[-0.4, 0.8],
                         help="Clip range for the commanded forward speed (m/s), mirrors this "
                              "env's own trained range.")
    parser.add_argument("--lateral-range", type=float, nargs=2, default=[-0.3, 0.3])
    parser.add_argument("--max-accel", type=float, default=1.5,
                         help="m/s^2: rate-limits the forward/lateral command before set_command "
                              "-- see module docstring, this env's QP goes infeasible under raw "
                              "step changes in a dynamically-recomputed goal-seeking command.")
    parser.add_argument("--steering", choices=["strafe", "turn"], default="strafe",
                         help="strafe (default): compute_nav_command's original speed+lateral "
                              "crab law, no yaw-rate used. turn: turn-to-face the goal bearing "
                              "then drive forward, exercising MPC's working yaw-rate tracking.")
    parser.add_argument("--kp-yaw", type=float, default=2.0,
                         help="Only used with --steering turn: commanded yaw-rate (rad/s) per "
                              "radian of heading error.")
    parser.add_argument("--yaw-rate-range", type=float, nargs=2, default=[-1.5, 1.5])
    parser.add_argument("--max-yaw-accel", type=float, default=3.0,
                         help="rad/s^2: rate-limits the yaw-rate command, same reason as "
                              "--max-accel. Only used with --steering turn.")
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--terrain-amplitude", type=float, default=None,
                         help="m peak-to-peak random bumps, fixed for this whole episode "
                              "(passed through as terrain_amplitude_range=(amp, amp)).")
    parser.add_argument("--terrain-footstep-planning", action="store_true",
                         help="Use the adopted terrain-aware footstep-planning fix (see "
                              "mpc/README.md) -- only matters if --terrain-amplitude is set.")
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--slowmo", type=float, default=1.0)
    args = parser.parse_args()

    if args.waypoint:
        route = [np.array(wp) for wp in args.waypoint]
    elif args.goal_x is not None and args.goal_y is not None:
        route = [np.array([args.goal_x, args.goal_y])]
    else:
        parser.error("give either --goal-x/--goal-y or at least one --waypoint")

    env = MPCResidualEnv(
        episode_seconds=args.seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        terrain_amplitude_range=(args.terrain_amplitude, args.terrain_amplitude)
            if args.terrain_amplitude is not None else None,
        terrain_footstep_planning=args.terrain_footstep_planning,
        seed=args.seed,
    )
    obs, _ = env.reset(seed=args.seed)
    speed_lo, speed_hi = args.speed_range
    lat_lo, lat_hi = args.lateral_range
    yaw_lo, yaw_hi = args.yaw_rate_range
    max_cmd_delta = args.max_accel / 200.0   # this env's CTRL_HZ is 200, not go1_env's 50
    max_yaw_delta = args.max_yaw_accel / 200.0

    max_steps = int(args.seconds * 200)
    n_wp = len(route)
    current_idx = 0
    goal = route[current_idx]
    x, y, yaw = 0.0, 0.0, 0.0
    cur_fwd, cur_lat, cur_yaw = 0.0, 0.0, 0.0
    frames = []
    reached, fell = False, False
    t_reached = None
    visits = []
    just_arrived_idx = None

    renderer = None
    if args.record:
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE   # wide fixed overhead view, not a tight tracking
        cam.lookat[:] = [0.0, 0.0, 0.0]          # cam -- a goal-directed path needs to stay in
        cam.distance = max(6.0, 1.3 * max(float(np.hypot(*wp)) for wp in route) + 2.0)
        cam.elevation = -70
        cam.azimuth = 90
        renderer = mj.Renderer(env.mujoco_go2.model, height=480, width=640)

    for step in range(1, max_steps + 1):
        dist = float(np.hypot(goal[0] - x, goal[1] - y))
        at_final_stop = (current_idx == n_wp - 1) and (not args.loop or n_wp == 1)
        if dist <= args.tolerance:
            if just_arrived_idx != current_idx:
                visits.append((current_idx, step / 200.0))
                just_arrived_idx = current_idx
            if at_final_stop:
                if not reached:
                    reached, t_reached = True, step / 200.0
            else:
                current_idx = (current_idx + 1) % n_wp
                goal = route[current_idx]
                just_arrived_idx = None

        if reached:
            target_fwd, target_lat, target_yaw = 0.0, 0.0, 0.0
        elif args.steering == "turn":
            target_fwd, target_yaw = compute_nav_command_turn(
                x, y, yaw, goal[0], goal[1], args.kp, args.kp_yaw, speed_lo, speed_hi,
                yaw_lo, yaw_hi)
            target_lat = 0.0
        else:
            target_fwd, target_lat = compute_nav_command(
                x, y, yaw, goal[0], goal[1], [], args.kp, speed_lo, speed_hi, lat_lo, lat_hi)
            target_yaw = 0.0
        cur_fwd += np.clip(target_fwd - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_lat - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_yaw - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        x, y = float(info["base_pos"][0]), float(info["base_pos"][1])
        yaw = float(info["yaw"])

        if args.record:
            if step % 8 == 0:
                renderer.update_scene(env.mujoco_go2.data, camera=cam)
                frames.append(renderer.render().copy())
        else:
            time.sleep(0.005)

        if terminated or truncated:
            fell = terminated
            break
        if reached and step > (t_reached * 200 + 200):
            break

    final_dist = float(np.hypot(goal[0] - x, goal[1] - y))
    if n_wp == 1:
        if fell:
            print(f"FELL/QP-FAILED before reaching the goal (final distance {final_dist:.2f} m).")
        elif reached:
            print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
                  f"final distance {final_dist:.2f} m.")
        else:
            print(f"Did NOT reach the goal within {args.seconds:.0f}s "
                  f"(final distance {final_dist:.2f} m, started {np.hypot(*route[0]):.2f} m away).")
    else:
        loop_s = " (looping)" if args.loop else ""
        print(f"Route{loop_s}: {len(visits)} waypoint arrival(s) across {n_wp}-waypoint route:")
        for idx, t in visits:
            print(f"  waypoint {idx} ({route[idx][0]:+.2f}, {route[idx][1]:+.2f}) reached at t={t:.1f}s")
        if fell:
            print(f"FELL/QP-FAILED (final distance {final_dist:.2f} m from waypoint {current_idx}).")
        elif reached:
            print(f"Completed the route, held the final waypoint for 1s after t={t_reached:.1f}s.")
        else:
            print(f"Did NOT complete the route within {args.seconds:.0f}s "
                  f"(currently targeting waypoint {current_idx}, {final_dist:.2f} m away).")

    if renderer is not None:
        renderer.close()   # see mpc/demo_gait_styles.py -- a stale GL context otherwise bleeds
                            # into the next Renderer created in the same process.
    if args.record and frames:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        base_fps = 200.0 / 8
        frame_duration_ms = (1000.0 / base_fps) / args.slowmo
        # optimize=True deliberately NOT used -- see mpc/demo_gait_styles.py, this Pillow
        # version corrupts/bloats multi-frame GIF output with it.
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=frame_duration_ms, loop=0)
        print(f"Saved {len(imgs)} frames to {args.record}")


if __name__ == "__main__":
    main()
