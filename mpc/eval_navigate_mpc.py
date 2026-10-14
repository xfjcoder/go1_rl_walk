"""
Stage 11 extension: randomized multi-episode evaluation of MPC-based point-to-goal navigation
(zero RL residual), mirroring eval_navigate.py's own metrics/methodology for a direct,
apples-to-apples comparison against pretrained/go2_latback's documented numbers (see
HISTORY.md's Stage 10a section: flat 100% success/0% falls/9.6s mean time-to-goal/0.65 path
efficiency, 32 episodes, goal distance 1.5-4m).

Single-process, sequential episodes (not a vectorized batch) -- same deliberate-simplicity choice
as eval_residual.py: MPCResidualEnv construction costs ~1.2s (CasADi/Pinocchio one-time setup),
acceptable for a one-off eval harness, not worth SubprocVecEnv's extra complexity here.

Usage:
    python3 eval_navigate_mpc.py --episodes 32
    python3 eval_navigate_mpc.py --episodes 32 --terrain-amplitude 0.0 0.08
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mpc_residual_env import MPCResidualEnv
from navigate import compute_nav_command
from navigate_mpc import compute_nav_command_turn


def run_episode(seed, goal_x, goal_y, seconds, kp, tolerance, speed_range, lateral_range,
                max_accel, terrain_amplitude, terrain_footstep_planning, steering="strafe",
                kp_yaw=2.0, yaw_rate_range=(-1.5, 1.5), max_yaw_accel=3.0):
    env = MPCResidualEnv(
        episode_seconds=seconds,
        speed_range=(0.0, 0.0), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        terrain_amplitude_range=(terrain_amplitude, terrain_amplitude)
            if terrain_amplitude is not None else None,
        terrain_footstep_planning=terrain_footstep_planning,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    speed_lo, speed_hi = speed_range
    lat_lo, lat_hi = lateral_range
    yaw_lo, yaw_hi = yaw_rate_range
    max_cmd_delta = max_accel / 200.0
    max_yaw_delta = max_yaw_accel / 200.0

    max_steps = int(seconds * 200)
    x, y, yaw = 0.0, 0.0, 0.0
    cur_fwd, cur_lat, cur_yaw = 0.0, 0.0, 0.0
    path_len = 0.0
    reached, t_reached = False, None
    fell, qp_failed_any = False, False

    for step in range(1, max_steps + 1):
        dist = float(np.hypot(goal_x - x, goal_y - y))
        if dist <= tolerance:
            if not reached:
                reached, t_reached = True, step / 200.0
            target_fwd, target_lat, target_yaw = 0.0, 0.0, 0.0
        elif steering == "turn":
            target_fwd, target_yaw = compute_nav_command_turn(
                x, y, yaw, goal_x, goal_y, kp, kp_yaw, speed_lo, speed_hi, yaw_lo, yaw_hi)
            target_lat = 0.0
        else:
            target_fwd, target_lat = compute_nav_command(
                x, y, yaw, goal_x, goal_y, [], kp, speed_lo, speed_hi, lat_lo, lat_hi)
            target_yaw = 0.0
        cur_fwd += np.clip(target_fwd - cur_fwd, -max_cmd_delta, max_cmd_delta)
        cur_lat += np.clip(target_lat - cur_lat, -max_cmd_delta, max_cmd_delta)
        cur_yaw += np.clip(target_yaw - cur_yaw, -max_yaw_delta, max_yaw_delta)
        env.set_command(cur_fwd, cur_lat, cur_yaw)

        obs, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        new_x, new_y = float(info["base_pos"][0]), float(info["base_pos"][1])
        path_len += float(np.hypot(new_x - x, new_y - y))
        x, y, yaw = new_x, new_y, float(info["yaw"])
        if info["qp_failed"]:
            qp_failed_any = True

        if terminated or truncated:
            fell = terminated
            break
        if reached and step > (t_reached * 200 + 200):   # linger 1s after reaching, then stop
            break

    final_dist = float(np.hypot(goal_x - x, goal_y - y))
    straight = float(np.hypot(goal_x, goal_y))
    return dict(reached=reached, fell=fell, qp_failed=qp_failed_any, t=t_reached,
                final_dist=final_dist, path_len=path_len, straight=straight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--goal-distance-range", type=float, nargs=2, default=[1.5, 4.0])
    ap.add_argument("--speed-range", type=float, nargs=2, default=[-0.4, 0.8])
    ap.add_argument("--lateral-range", type=float, nargs=2, default=[-0.3, 0.3])
    ap.add_argument("--max-accel", type=float, default=1.5)
    ap.add_argument("--steering", choices=["strafe", "turn"], default="strafe")
    ap.add_argument("--kp-yaw", type=float, default=2.0)
    ap.add_argument("--yaw-rate-range", type=float, nargs=2, default=[-1.5, 1.5])
    ap.add_argument("--max-yaw-accel", type=float, default=3.0)
    ap.add_argument("--terrain-amplitude", type=float, nargs="+", default=[None],
                     help="terrain amplitude(s) in m (peak-to-peak) to evaluate at. "
                          "Default: flat ground only.")
    ap.add_argument("--terrain-footstep-planning", action="store_true")
    ap.add_argument("--seed0", type=int, default=20_000)
    args = ap.parse_args()

    print(f"episodes={args.episodes}  goal distance {args.goal_distance_range[0]}-"
          f"{args.goal_distance_range[1]} m, bearing 0-360deg, kp={args.kp}, "
          f"tolerance={args.tolerance} m, max_accel={args.max_accel} m/s^2")

    for amp in args.terrain_amplitude:
        rng = np.random.default_rng(args.seed0)
        bearing = rng.uniform(0.0, 2 * np.pi, args.episodes)
        dist0 = rng.uniform(args.goal_distance_range[0], args.goal_distance_range[1], args.episodes)
        goal_x, goal_y = dist0 * np.cos(bearing), dist0 * np.sin(bearing)

        results = [run_episode(args.seed0 + i, float(goal_x[i]), float(goal_y[i]), args.seconds,
                                args.kp, args.tolerance, tuple(args.speed_range),
                                tuple(args.lateral_range), args.max_accel, amp,
                                args.terrain_footstep_planning, args.steering, args.kp_yaw,
                                tuple(args.yaw_rate_range), args.max_yaw_accel)
                   for i in range(args.episodes)]

        reached = np.array([r["reached"] for r in results])
        fell = np.array([r["fell"] for r in results])
        qp_failed = np.array([r["qp_failed"] for r in results])
        t_reach = np.array([r["t"] for r in results if r["reached"]])
        eff = np.array([r["straight"] / r["path_len"] for r in results
                         if r["reached"] and r["path_len"] > 1e-6])

        terr = "flat" if amp is None else f"terrain {amp * 100:.0f} cm"
        print(f"{terr:>13s}: success_rate={reached.mean():.2f}  fall_rate={fell.mean():.2f}  "
              f"qp_failed_rate={qp_failed.mean():.2f}  "
              f"time_to_goal={t_reach.mean() if t_reach.size else float('nan'):5.1f}s  "
              f"path_efficiency={eff.mean() if eff.size else float('nan'):.2f}", flush=True)


if __name__ == "__main__":
    main()
