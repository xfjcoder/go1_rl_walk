"""
Stage 11 step 5: randomized multi-seed evaluation of the trained residual policy against
zero-residual (pure MPC), mirroring this project's own evaluation discipline (see memory:
go1-eval-method -- judge changes over randomized multi-seed trials, not single runs).

Each episode samples a random command (speed/lateral/yaw-rate) and random domain
parameters (friction/mass scale) from the SAME training ranges, with the SAME seed used
for both arms (trained vs. zero-residual) so the comparison is paired, not just two
independent random samples that happen to differ.

Single-process, sequential episodes (not SubprocVecEnv) -- deliberately simple for this
evaluation harness: ~7 minutes per arm at this episode count, not worth the added
multiprocessing complexity (and this stage already spent real effort debugging a
multiprocessing-specific crash, see mpc/README.md) for a one-off evaluation script.

Usage:
    python3 mpc/eval_residual.py --run-dir runs/v1 --episodes 32
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpc_residual_env import MPCResidualEnv


def run_episodes(model, vecnorm_path, env_kwargs, episodes, seed0, use_residual):
    """Runs `episodes` episodes, one at a time (fresh env each episode, matching the
    project's own per-episode reset discipline), returning a dict of per-episode arrays."""
    results = {"survived": [], "ep_len": [], "speed": [], "lateral": [], "yaw_rate": [],
               "effort": [], "qp_failed": [], "fell": []}

    for i in range(episodes):
        env = DummyVecEnv([lambda: MPCResidualEnv(**env_kwargs, seed=seed0 + i)])
        if use_residual:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
        obs = env.reset()

        ep_speed, ep_lateral, ep_yaw_rate, ep_effort = [], [], [], []
        qp_failed_any = False
        fell = False
        steps = 0
        while True:
            if use_residual:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros((1, 12), dtype=np.float32)
            obs, reward, done, info = env.step(action)
            steps += 1
            comp = info[0]["reward_components"]
            ep_speed.append(comp["speed"])
            ep_lateral.append(comp["lateral"])
            ep_yaw_rate.append(comp["yaw_rate"])
            ep_effort.append(comp["effort"])
            if info[0].get("qp_failed"):
                qp_failed_any = True
            if done[0]:
                fell = not info[0].get("TimeLimit.truncated", False)
                break
        env.close()

        results["survived"].append(not fell)
        results["ep_len"].append(steps)
        results["speed"].append(np.mean(ep_speed))
        results["lateral"].append(np.mean(ep_lateral))
        results["yaw_rate"].append(np.mean(ep_yaw_rate))
        results["effort"].append(np.mean(ep_effort))
        results["qp_failed"].append(qp_failed_any)
        results["fell"].append(fell)

    return {k: np.array(v) for k, v in results.items()}


def summarize(label, r, max_steps):
    print(f"{label:>14s}: success_rate={np.mean(r['survived']):.2f}  "
          f"fall_rate={np.mean(r['fell']):.2f}  "
          f"qp_failed_rate={np.mean(r['qp_failed']):.2f}  "
          f"ep_len_mean={np.mean(r['ep_len']):6.1f}/{max_steps}  "
          f"speed_track={np.mean(r['speed']):.3f}  lateral_track={np.mean(r['lateral']):.3f}  "
          f"yaw_rate_track={np.mean(r['yaw_rate']):.3f}  effort={np.mean(r['effort']):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default="runs/v1")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=50_000)
    ap.add_argument("--episode-seconds", type=float, default=10.0)
    ap.add_argument("--speed-range", type=float, nargs=2, default=[-0.4, 0.8])
    ap.add_argument("--lateral-range", type=float, nargs=2, default=[-0.3, 0.3])
    ap.add_argument("--yaw-rate-range", type=float, nargs=2, default=[-1.0, 1.0])
    ap.add_argument("--friction-range", type=float, nargs=2, default=[0.6, 1.1])
    ap.add_argument("--mass-scale-range", type=float, nargs=2, default=[0.8, 1.2])
    ap.add_argument("--push-velocity", type=float, default=0.0)
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent / args.run_dir
    model = PPO.load(str(run_dir / "checkpoints" / "final.zip"))
    vecnorm_path = str(run_dir / "checkpoints" / "vecnormalize_final.pkl")

    env_kwargs = dict(
        episode_seconds=args.episode_seconds,
        speed_range=tuple(args.speed_range),
        lateral_range=tuple(args.lateral_range),
        yaw_rate_range=tuple(args.yaw_rate_range),
        friction_range=tuple(args.friction_range),
        mass_scale_range=tuple(args.mass_scale_range),
        push_velocity=args.push_velocity,
    )
    max_steps = int(args.episode_seconds * 200)

    print(f"model={run_dir}/checkpoints/final.zip  episodes={args.episodes}  "
          f"speed={args.speed_range}  lateral={args.lateral_range}  yaw_rate={args.yaw_rate_range}  "
          f"friction={args.friction_range}  mass_scale={args.mass_scale_range}")

    print("Running zero-residual (pure MPC) arm...")
    zero_results = run_episodes(model, vecnorm_path, env_kwargs, args.episodes, args.seed0, use_residual=False)
    print("Running trained-residual arm...")
    trained_results = run_episodes(model, vecnorm_path, env_kwargs, args.episodes, args.seed0, use_residual=True)

    summarize("zero-residual", zero_results, max_steps)
    summarize("trained", trained_results, max_steps)


if __name__ == "__main__":
    main()
