"""
Stage 10a: drive a trained walking policy toward a world-frame (x, y) goal, with NO retraining.

Why this works at all: the reward (`r_heading` in envs/go1_env.py) actively penalizes any yaw
deviation from 0, so a converged policy holds its heading near world-+x throughout an episode --
confirmed directly in envs/go1_env.py, not assumed. World frame therefore stays close enough to
body frame that a goal vector can be fed straight in as a (target_speed, target_lateral_speed)
command, using pretrained/go2_latback exactly as trained (it already supports forward, backward,
and sideways walking simultaneously). No new mechanism, no fine-tuning -- just a thin outer loop:
a proportional controller recomputes the command every control step from the live vector to the
goal (rotated into the robot's current frame using its measured yaw, so a residual heading error
doesn't silently bias the command), decelerating smoothly as the goal nears and stopping within
--tolerance.

Deliberately NOT attempted here (see HISTORY.md / memory for the full Stage 10 roadmap): obstacle
avoidance (10b) and genuine turning (10c) -- this script only answers "can the existing checkpoint
reach an arbitrary flat-ground point ahead/behind/beside it," nothing more.

Usage:
    python navigate.py --goal-x 3.0 --goal-y 1.0 --record out.gif
    python navigate.py --goal-x -2.0 --goal-y 0.0 --terrain-amplitude 0.08 --record out.gif
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="pretrained/go2_latback",
                         help="A checkpoint that supports lateral/backward commands (the default, "
                              "go2_latback) -- go2_gaitclock has no lateral command dimension to drive.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--goal-x", type=float, required=True, help="Goal position, world frame, m.")
    parser.add_argument("--goal-y", type=float, required=True)
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
    parser.add_argument("--record", type=str, default=None, help="Save an offscreen-rendered GIF.")
    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--camera", type=str, default="track", choices=["track", "chase_rear", "topdown"])
    args = parser.parse_args()

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

    max_steps = int(args.seconds * 50)
    goal = np.array([args.goal_x, args.goal_y])
    x, y, yaw = 0.0, 0.0, 0.0   # domain_randomize=False -> the keyframe's exact start pose, no jitter
    frames = []
    reached, fell = False, False
    t_reached = None

    for step in range(1, max_steps + 1):
        dx, dy = goal[0] - x, goal[1] - y
        dist = float(np.hypot(dx, dy))
        if dist <= args.tolerance:
            if not reached:            # latch the FIRST time only -- otherwise t_reached keeps
                reached, t_reached = True, step / 50.0   # re-stamping to "now" every step the
            raw_env.set_command(0.0, 0.0)                # robot stays near the goal, and the
                                                          # linger-then-stop check below never fires
        else:
            c, s = np.cos(yaw), np.sin(yaw)
            fwd_err, lat_err = dx * c + dy * s, -dx * s + dy * c    # world vector -> robot's own frame
            fwd_cmd = float(np.clip(args.kp * fwd_err, speed_lo, speed_hi))
            lat_cmd = float(np.clip(args.kp * lat_err, lat_lo, lat_hi))
            raw_env.set_command(fwd_cmd, lat_cmd)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        x, y = float(info[0]["base_pos"][0]), float(info[0]["base_pos"][1])
        yaw = float(info[0]["yaw"])

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
    if fell:
        print(f"FELL before reaching the goal (final distance {final_dist:.2f} m).")
    elif reached:
        print(f"Reached goal ({goal[0]:.2f}, {goal[1]:.2f}) in {t_reached:.1f}s, "
              f"final distance {final_dist:.2f} m.")
    else:
        print(f"Did NOT reach the goal within {args.seconds:.0f}s "
              f"(final distance {final_dist:.2f} m, started {np.hypot(*goal):.2f} m away).")

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
