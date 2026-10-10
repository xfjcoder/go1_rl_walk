"""
Stage 10a/10b: drive a trained walking policy toward a world-frame (x, y) goal, with NO
retraining, optionally steering around static obstacles along the way.

Why this works at all: the reward (`r_heading` in envs/go1_env.py) actively penalizes any yaw
deviation from 0, so a converged policy holds its heading near world-+x throughout an episode --
confirmed directly in envs/go1_env.py, not assumed. World frame therefore stays close enough to
body frame that a goal vector can be fed straight in as a (target_speed, target_lateral_speed)
command, using pretrained/go2_latback exactly as trained (it already supports forward, backward,
and sideways walking simultaneously). No new mechanism, no fine-tuning -- just a thin outer loop:
a proportional controller (see compute_nav_command below) recomputes the command every control
step from the live vector to the goal (rotated into the robot's current frame using its measured
yaw, so a residual heading error doesn't silently bias the command) plus a repulsive term from any
nearby obstacles (--nav-obstacles; ground-truth positions, queried from the env, not sensed),
decelerating smoothly as the goal nears and stopping within --tolerance.

Deliberately NOT attempted here (see HISTORY.md / memory for the full Stage 10 roadmap): genuine
turning (10c, the still-unused yaw-rate command slot) and perception-based (rather than
ground-truth) obstacle detection.

Usage:
    python navigate.py --goal-x 3.0 --goal-y 1.0 --record out.gif
    python navigate.py --goal-x -2.0 --goal-y 0.0 --terrain-amplitude 0.08 --record out.gif
    python navigate.py --goal-x 4.0 --goal-y 0.0 --nav-obstacles 4 --record out.gif
    python navigate.py --waypoint 2 0 --waypoint 2 2 --waypoint 0 2 --waypoint 0 0 --loop \
        --seconds 60 --record patrol.gif
"""
import argparse
import json
import os
import re
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv
from eval_policy import latest_checkpoint


def compute_nav_command(x, y, yaw, goal_x, goal_y, obstacles, kp, speed_lo, speed_hi, lat_lo, lat_hi,
                         obstacle_gain=1.5, obstacle_influence=0.8):
    """Proportional goal-attraction + linear obstacle-repulsion steering law (a simplified
    artificial potential field -- linear falloff instead of the textbook 1/d^2 term, so there's no
    singularity to guard against as the robot nears an obstacle's surface). Computed in world
    frame, then rotated into the robot's own frame via its measured yaw (not assumed zero) and
    clipped per-axis to the checkpoint's trained command range.

    obstacles: list of (ox, oy, radius) tuples -- ground truth, meant to come straight from
    Go1FlatEnv.obstacle_positions (+ its shared nav_obstacle_radius), not a sensed estimate.
    Scalar-only (one robot at a time) -- eval_navigate.py's parallel episodes call this once per
    episode index in a loop, since each one has its own independently-randomized obstacle set."""
    wx, wy = goal_x - x, goal_y - y   # attractive component: straight toward the goal
    for ox, oy, r in obstacles:
        ex, ey = x - ox, y - oy       # vector FROM the obstacle's centre TO the robot
        center_dist = max(float(np.hypot(ex, ey)), 1e-3)
        clearance = center_dist - r   # distance to the obstacle's own surface, not its centre
        if clearance < obstacle_influence:
            strength = obstacle_gain * (obstacle_influence - clearance) / obstacle_influence
            wx += strength * ex / center_dist
            wy += strength * ey / center_dist
    c, s = np.cos(yaw), np.sin(yaw)
    fwd_err, lat_err = wx * c + wy * s, -wx * s + wy * c    # world vector -> robot's own frame
    fwd_cmd = float(np.clip(kp * fwd_err, speed_lo, speed_hi))
    lat_cmd = float(np.clip(kp * lat_err, lat_lo, lat_hi))
    return fwd_cmd, lat_cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="pretrained/go2_latback",
                         help="A checkpoint that supports lateral/backward commands (the default, "
                              "go2_latback) -- go2_gaitclock has no lateral command dimension to drive.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--goal-x", type=float, default=None, help="Goal position, world frame, m. "
                         "Ignored if --waypoint is given at all.")
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--waypoint", type=float, nargs=2, action="append", default=None,
                         metavar=("X", "Y"),
                         help="Add a waypoint (repeatable, visited in the order given) to a patrol "
                              "route instead of a single goal -- e.g. "
                              "'--waypoint 2 0 --waypoint 2 2 --waypoint 0 2' for a 3-stop route. "
                              "Reuses the exact same per-step steering law for each leg; the only "
                              "new behavior is switching targets on arrival instead of stopping.")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=False,
                         help="With --waypoint: cycle back to the first waypoint after the last "
                              "instead of stopping there. Ignored in single-goal mode.")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Success radius, m.")
    parser.add_argument("--kp", type=float, default=1.0,
                         help="Proportional gain: commanded speed (m/s) per metre of remaining "
                              "goal-vector component, clipped to the checkpoint's trained speed range.")
    parser.add_argument("--seconds", type=float, default=25.0, help="Episode time budget.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--terrain-amplitude", type=float, default=None,
                         help="Navigate across rough terrain of this amplitude (m, peak-to-peak). "
                              "Stage 10a is scoped to flat/rough terrain only -- slope/stairs form a "
                              "ridge along world-y, so a goal crossing one at an angle is untested.")
    parser.add_argument("--nav-obstacles", type=int, default=0,
                         help="Scatter this many static, impassable cylinder obstacles in the "
                              "course (Stage 10b) and steer around them. 0 (default) = none.")
    parser.add_argument("--nav-obstacle-radius", type=float, default=0.15)
    parser.add_argument("--nav-obstacle-height", type=float, default=0.5)
    parser.add_argument("--obstacle-avoid-gain", type=float, default=1.5,
                         help="Repulsive-field strength; only matters with --nav-obstacles > 0.")
    parser.add_argument("--obstacle-influence-radius", type=float, default=0.8,
                         help="Distance (m, from an obstacle's surface) at which its repulsion "
                              "starts being felt at all; only matters with --nav-obstacles > 0.")
    parser.add_argument("--record", type=str, default=None, help="Save an offscreen-rendered GIF.")
    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--camera", type=str, default="track", choices=["track", "chase_rear", "topdown"])
    args = parser.parse_args()

    if args.waypoint:
        route = [np.array(wp) for wp in args.waypoint]
    elif args.goal_x is not None and args.goal_y is not None:
        route = [np.array([args.goal_x, args.goal_y])]
    else:
        parser.error("give either --goal-x/--goal-y or at least one --waypoint")

    render_mode = "rgb_array" if args.record else "human"

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    model_path = os.path.join(ckpt_dir, args.model) if args.model else latest_checkpoint(ckpt_dir)
    m = re.search(r"go1_flat_(\d+)_steps$", os.path.basename(model_path))
    vecnorm_path = os.path.join(ckpt_dir, f"go1_flat_vecnormalize_{m.group(1)}_steps.pkl") if m \
        else os.path.join(ckpt_dir, "vecnormalize_final.pkl")
    with open(os.path.join(args.run_dir, "env_kwargs.json")) as f:
        env_kwargs = json.load(f)

    # Capture the checkpoint's own trained command range for clipping BEFORE overwriting it below
    # (same pattern as play.py: command_speed_range/lateral_speed_range -> None means "don't
    # resample every reset", not "no limit" -- the actual limits are these saved values).
    speed_lo, speed_hi = env_kwargs.get("command_speed_range") or (env_kwargs["target_speed"],) * 2
    lat_lo, lat_hi = env_kwargs.get("lateral_speed_range") or (env_kwargs.get("target_lateral_speed", 0.0),) * 2

    env_kwargs["max_episode_seconds"] = args.seconds   # the env's own default (20s) isn't saved in
                                                        # env_kwargs.json -- align it with our budget
                                                        # so a distant goal isn't cut off early.
    env_kwargs["command_speed_range"] = None
    env_kwargs["lateral_speed_range"] = None
    env_kwargs["target_speed"] = 0.0
    env_kwargs["target_lateral_speed"] = 0.0
    env_kwargs["terrain_amplitude_range"] = None
    env_kwargs["terrain_amplitude"] = args.terrain_amplitude
    env_kwargs["slope_range"] = None
    env_kwargs["slope_deg"] = None
    env_kwargs["stair_height_range"] = None
    env_kwargs["stair_height"] = None
    env_kwargs["obstacle_height_range"] = None
    env_kwargs["obstacle_height"] = None
    env_kwargs["nav_obstacles"] = args.nav_obstacles
    env_kwargs["nav_obstacle_radius"] = args.nav_obstacle_radius
    env_kwargs["nav_obstacle_height"] = args.nav_obstacle_height

    def make_env():
        return Go1FlatEnv(render_mode=render_mode, domain_randomize=False, camera=args.camera, **env_kwargs)

    env = DummyVecEnv([make_env])
    try:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("No VecNormalize stats found, running without observation normalization.")

    model = PPO.load(model_path)
    raw_env = env.envs[0]   # VecNormalize forwards .envs via __getattr__, same as play.py's own usage

    if args.seed is not None:
        env.seed(args.seed)
    obs = env.reset()
    obstacles = [(ox, oy, raw_env.nav_obstacle_radius) for ox, oy in raw_env.obstacle_positions]
    if obstacles:
        print(f"obstacles (ground truth): {[(round(o[0], 2), round(o[1], 2)) for o in obstacles]}")

    max_steps = int(args.seconds * 50)
    n_wp = len(route)
    current_idx = 0
    goal = route[current_idx]
    x, y, yaw = 0.0, 0.0, 0.0   # domain_randomize=False -> the keyframe's exact start pose, no jitter
    frames = []
    reached, fell = False, False
    t_reached = None
    closest_clearance = float("inf")
    visits = []            # (waypoint_idx, time_s) in the order each was reached, incl. repeats if looping
    just_arrived_idx = None   # guards against re-logging the same arrival every step while lingering

    for step in range(1, max_steps + 1):
        dist = float(np.hypot(goal[0] - x, goal[1] - y))
        # n_wp==1 always "stops" regardless of --loop (nothing else to cycle to -- reduces this whole
        # block to the exact single-goal behavior byte-for-byte when no --waypoint was given at all).
        at_final_stop = (current_idx == n_wp - 1) and (not args.loop or n_wp == 1)
        if dist <= args.tolerance:
            if just_arrived_idx != current_idx:
                visits.append((current_idx, step / 50.0))
                just_arrived_idx = current_idx
            if at_final_stop:
                if not reached:        # latch the FIRST time only -- otherwise t_reached keeps
                    reached, t_reached = True, step / 50.0   # re-stamping to "now" every step the
            else:                      # robot stays near it, and the linger-then-stop check below
                current_idx = (current_idx + 1) % n_wp        # never fires
                goal = route[current_idx]
                just_arrived_idx = None   # so the NEXT waypoint's own arrival gets logged too

        if reached:
            raw_env.set_command(0.0, 0.0)
        else:
            fwd_cmd, lat_cmd = compute_nav_command(
                x, y, yaw, goal[0], goal[1], obstacles, args.kp, speed_lo, speed_hi, lat_lo, lat_hi,
                args.obstacle_avoid_gain, args.obstacle_influence_radius)
            raw_env.set_command(fwd_cmd, lat_cmd)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        x, y = float(info[0]["base_pos"][0]), float(info[0]["base_pos"][1])
        yaw = float(info[0]["yaw"])
        for ox, oy, r in obstacles:
            closest_clearance = min(closest_clearance, float(np.hypot(x - ox, y - oy)) - r)

        if args.record:
            frame = raw_env.render(camera=args.camera)
            if frame is not None:
                frames.append(frame)
        else:
            time.sleep(0.02)

        if done[0]:
            fell = not info[0].get("TimeLimit.truncated", False)
            break
        if reached and step > (t_reached * 50 + 50):   # linger 1s after reaching, then stop
            break

    final_dist = float(np.hypot(goal[0] - x, goal[1] - y))
    if n_wp == 1:
        # exact original single-goal wording, unchanged
        if fell:
            print(f"FELL before reaching the goal (final distance {final_dist:.2f} m).")
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
            print(f"FELL (final distance {final_dist:.2f} m from waypoint {current_idx}).")
        elif reached:
            print(f"Completed the route, held the final waypoint for 1s after t={t_reached:.1f}s.")
        elif args.loop:
            print(f"Time budget ({args.seconds:.0f}s) ran out mid-loop "
                  f"(currently targeting waypoint {current_idx}, {final_dist:.2f} m away).")
        else:
            print(f"Did NOT complete the route within {args.seconds:.0f}s "
                  f"(currently targeting waypoint {current_idx}, {final_dist:.2f} m away).")
    if obstacles:
        print(f"Closest approach to any obstacle's surface: {closest_clearance:.2f} m "
              f"({'collision' if closest_clearance < 0 else 'clear'}).")

    env.close()
    if args.record and frames:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames[::args.frame_stride]]
        base_fps = 50.0 / args.frame_stride
        frame_duration_ms = (1000.0 / base_fps) / args.slowmo
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=frame_duration_ms, loop=0)
        print(f"Saved {len(imgs)} frames to {args.record}")


if __name__ == "__main__":
    main()
