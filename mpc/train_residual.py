"""
Stage 11 step 4: train the RL-residual policy on top of the (unmodified) MPC controller.

Mirrors this project's own train.py conventions where they transfer directly (PPO
hyperparameters, VecNormalize, linear LR decay, CheckpointCallback, per-run
args.json/env_kwargs.json for reproducibility, RewardComponentLoggingCallback reused
unmodified since MPCResidualEnv's info["reward_components"] matches its expected shape)
-- intentionally narrower than train.py itself (no curriculum ramps) since this is a
first exploratory run, not a mature multi-month training pipeline. --auto-resume exists
specifically because of an observed intermittent crash (see mpc/README.md) -- checkpoints
save every iteration so a crash loses at most one iteration's worth of progress.

Usage:
    python3 mpc/train_residual.py --run-name v1 --timesteps 2000000
    python3 mpc/train_residual.py --run-name v1 --timesteps 2000000 --auto-resume
    tensorboard --logdir mpc/runs
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpc_residual_env import MPCResidualEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import RewardComponentLoggingCallback, linear_schedule


def make_env(rank: int, seed: int, env_kwargs: dict):
    def _init():
        env = MPCResidualEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def vecnormalize_path_for(checkpoint_zip: str):
    """Matching VecNormalize stats file for a checkpoint zip saved by CheckpointCallback
    (save_vecnormalize=True) with name_prefix="mpc_residual" -- mirrors train.py's own
    vecnormalize_path_for exactly, same naming convention."""
    d, base = os.path.split(checkpoint_zip)
    m = re.match(r"mpc_residual_(\d+)_steps\.zip$", base)
    if m:
        return os.path.join(d, f"mpc_residual_vecnormalize_{m.group(1)}_steps.pkl")
    return None


def latest_checkpoint(ckpt_dir: str):
    """Most recent mpc_residual_<N>_steps.zip in ckpt_dir by N, or None if none exist yet."""
    candidates = glob.glob(os.path.join(ckpt_dir, "mpc_residual_*_steps.zip"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(re.search(r"_(\d+)_steps\.zip$", p).group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True,
                     help="Writes to mpc/runs/<name>/{checkpoints,logs} + args.json/env_kwargs.json.")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual-scale", type=float, default=5.0)
    ap.add_argument("--residual-effort-weight", type=float, default=0.002)
    ap.add_argument("--episode-seconds", type=float, default=10.0)
    ap.add_argument("--speed-range", type=float, nargs=2, default=[-0.4, 0.8])
    ap.add_argument("--lateral-range", type=float, nargs=2, default=[-0.3, 0.3])
    ap.add_argument("--yaw-rate-range", type=float, nargs=2, default=[-1.0, 1.0])
    ap.add_argument("--friction-range", type=float, nargs=2, default=[0.6, 1.1])
    ap.add_argument("--mass-scale-range", type=float, nargs=2, default=[0.8, 1.2])
    ap.add_argument("--push-velocity", type=float, default=0.0)
    ap.add_argument("--auto-resume", action="store_true",
                     help="If mpc/runs/<name>/checkpoints already has a checkpoint, resume from "
                          "the latest one instead of starting fresh -- --timesteps is interpreted "
                          "as the TARGET TOTAL (remaining = timesteps - checkpoint's own "
                          "num_timesteps), computed fresh each call so repeated resumes don't "
                          "dilute the LR schedule (see HISTORY.md Stage 10c for why that matters).")
    args = ap.parse_args()

    buffer_size = args.n_steps * args.n_envs
    if buffer_size % args.batch_size != 0:
        candidates = [d for d in range(args.batch_size, 0, -1) if buffer_size % d == 0]
        fixed = candidates[0] if candidates else buffer_size
        print(f"WARNING: batch_size={args.batch_size} does not divide rollout buffer "
              f"size ({args.n_steps} x {args.n_envs} = {buffer_size}). Using batch_size={fixed}.")
        args.batch_size = fixed

    run_dir = Path(__file__).resolve().parent / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        residual_scale=args.residual_scale,
        residual_effort_weight=args.residual_effort_weight,
        speed_range=tuple(args.speed_range),
        lateral_range=tuple(args.lateral_range),
        yaw_rate_range=tuple(args.yaw_rate_range),
        friction_range=tuple(args.friction_range),
        mass_scale_range=tuple(args.mass_scale_range),
        push_velocity=args.push_velocity,
        episode_seconds=args.episode_seconds,
    )
    with open(run_dir / "env_kwargs.json", "w") as f:
        json.dump(env_kwargs, f, indent=2)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # start_method="spawn", not the Linux default "fork": CasADi/Pinocchio/OSQP are native
    # libraries that aren't guaranteed fork-safe (a forked child only duplicates the calling
    # thread, not any internal thread pools/mutexes those libraries may hold) -- confirmed
    # this matters here, "fork" crashed a worker (EOFError on the parent's recv) under load.
    env = SubprocVecEnv([make_env(i, args.seed, env_kwargs) for i in range(args.n_envs)],
                        start_method="spawn")
    env = VecMonitor(env)

    resume_ckpt = latest_checkpoint(str(ckpt_dir)) if args.auto_resume else None
    if resume_ckpt:
        vec_path = vecnormalize_path_for(resume_ckpt)
        env = VecNormalize.load(vec_path, env)
        env.training = True
        model = PPO.load(resume_ckpt, env=env, tensorboard_log=str(log_dir))
        remaining = max(args.timesteps - model.num_timesteps, 0)
        print(f"Resuming from {resume_ckpt} (already {model.num_timesteps} steps); "
              f"{remaining} remaining toward target {args.timesteps}.")
        if remaining == 0:
            print("Target already reached, nothing to do.")
            return
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(log_dir),
            verbose=1,
            seed=args.seed,
        )
        remaining = args.timesteps

    # save_freq counts CALLBACK calls (one per vectorized step, i.e. n_envs timesteps each), not
    # raw total timesteps -- save_freq=args.n_steps means "save every iteration" (n_steps*n_envs
    # total timesteps between saves). Far more frequent than train.py's own 200_000-step cadence:
    # this is a first exploratory run on a controller with an observed intermittent crash (see
    # mpc/README.md), so minimizing progress lost per crash matters more here than it does for
    # train.py's own much longer, much more stable runs.
    checkpoint_callback = CheckpointCallback(
        save_freq=args.n_steps,
        save_path=str(ckpt_dir),
        name_prefix="mpc_residual",
        save_vecnormalize=True,
    )
    reward_logger = RewardComponentLoggingCallback()

    model.learn(total_timesteps=remaining, reset_num_timesteps=(resume_ckpt is None),
                callback=[checkpoint_callback, reward_logger])

    model.save(str(ckpt_dir / "final"))
    env.save(str(ckpt_dir / "vecnormalize_final.pkl"))
    print(f"Done. Checkpoints in {ckpt_dir}, tensorboard logs in {log_dir}")


if __name__ == "__main__":
    main()
