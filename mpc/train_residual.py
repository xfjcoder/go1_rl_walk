"""
Stage 11 step 4: train the RL-residual policy on top of the (unmodified) MPC controller.

Mirrors this project's own train.py conventions where they transfer directly (PPO
hyperparameters, VecNormalize, linear LR decay, CheckpointCallback, per-run
args.json/env_kwargs.json for reproducibility, RewardComponentLoggingCallback reused
unmodified since MPCResidualEnv's info["reward_components"] matches its expected shape).
--auto-resume exists specifically because of an observed intermittent crash (see
mpc/README.md) -- checkpoints save every iteration so a crash loses at most one
iteration's worth of progress.

Terrain curriculum ramps (--terrain-curriculum-start/-steps etc.) mirror train.py's own
RampCallback pattern, reused unmodified, but are opt-in here: unlike train.py (where
curriculum is on by default), omitting these flags samples uniformly across the full
--terrain-amplitude-range/--slope-deg-range/--stair-height-range from step 0, reproducing
terrain_v1's exact prior behavior byte-for-byte. Pass them to replicate the staged
terrain->hard-mining recipe HISTORY.md's pure-RL pipeline used (ramp amplitude up over
most of training, then a separate --auto-resume invocation with a narrowed, harder-only
range to hard-mine the worst corner -- same two-pass idea as train.py's own
--terrain-amp-min hard-mining flag).

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
from train import RewardComponentLoggingCallback, RampCallback, linear_schedule


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
    if base == "final.zip":  # model.save(ckpt_dir / "final") at the end of a run -- only
        return os.path.join(d, "vecnormalize_final.pkl")  # reachable via --resume (an explicit
        # path), never --auto-resume (latest_checkpoint() only globs mpc_residual_*_steps.zip).
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
    ap.add_argument("--terrain-amplitude-range", type=float, nargs=2, default=None,
                     help="(lo, hi) m peak-to-peak random bumps, sampled each reset. "
                          "Omit (default) for flat ground.")
    ap.add_argument("--slope-deg-range", type=float, nargs=2, default=None,
                     help="(lo, hi) degrees, sampled each reset; uphill/downhill 50/50.")
    ap.add_argument("--stair-height-range", type=float, nargs=2, default=None,
                     help="(lo, hi) m per riser, sampled each reset; ascending/descending 50/50.")
    ap.add_argument("--use-terrain-heightmap", action="store_true",
                     help="Add a 3x3 local terrain heightmap (9 dims) ahead of the trunk to the "
                          "observation (obs_dim 47->56) -- mirrors envs/go1_env.py's own "
                          "--use-terrain-heightmap, needed there to handle its hardest terrain "
                          "corners since pure proprioception can't see a slope/stair coming.")
    ap.add_argument("--terrain-footstep-planning", action="store_true",
                     help="Make MPC's OWN footstep planner terrain-aware (TerrainAwareGait in "
                          "mpc_residual_env.py), instead of the residual being the only terrain-"
                          "aware thing (that's --use-terrain-heightmap, a separate flag). Fixes "
                          "MPC's own touchdown-height computation, hardcoded flat in the vendored "
                          "submodule.")
    ap.add_argument("--terrain-friction-cone", action="store_true",
                     help="Orient the MPC's QP friction-cone constraints around the real local "
                          "terrain normal instead of always assuming world-up (TerrainAware"
                          "CentroidalMPC in mpc_residual_env.py). Independent of "
                          "--terrain-footstep-planning (that fixes where feet land; this fixes "
                          "what forces are assumed achievable once planted).")
    ap.add_argument("--terrain-foothold-optimization", action="store_true",
                     help="Search nearby (x, y) candidate footholds for a flatter spot instead of "
                          "only correcting the nominal Raibert point's height (TerrainAwareGait."
                          "_best_foothold_xy in mpc_residual_env.py). Requires "
                          "--terrain-footstep-planning; a no-op otherwise.")
    ap.add_argument("--gait-style", type=str, default="trot",
                     choices=["trot", "pace", "bound", "walk"],
                     help="Contact-schedule parametrization MPC's Gait uses (GAIT_STYLE_PRESETS "
                          "in mpc_residual_env.py) -- a classical, non-learned choice, not "
                          "something the residual controls. 'trot' (default) reproduces every "
                          "prior result exactly.")
    ap.add_argument("--terrain-curriculum-start", type=float, default=None,
                     help="If given (needs --terrain-amplitude-range), ramp the sampled upper "
                          "bound from this value up to --terrain-amplitude-range's own hi over "
                          "--terrain-curriculum-steps, instead of sampling the full range from "
                          "step 0. Counted from THIS run's start, so also works across --auto-resume.")
    ap.add_argument("--terrain-curriculum-steps", type=int, default=500_000)
    ap.add_argument("--slope-curriculum-start", type=float, default=None,
                     help="Same idea as --terrain-curriculum-start, for --slope-deg-range.")
    ap.add_argument("--slope-curriculum-steps", type=int, default=500_000)
    ap.add_argument("--stair-curriculum-start", type=float, default=None,
                     help="Same idea as --terrain-curriculum-start, for --stair-height-range.")
    ap.add_argument("--stair-curriculum-steps", type=int, default=500_000)
    ap.add_argument("--auto-resume", action="store_true",
                     help="If mpc/runs/<name>/checkpoints already has a checkpoint, resume from "
                          "the latest one instead of starting fresh -- --timesteps is interpreted "
                          "as the TARGET TOTAL (remaining = timesteps - checkpoint's own "
                          "num_timesteps), computed fresh each call so repeated resumes don't "
                          "dilute the LR schedule (see HISTORY.md Stage 10c for why that matters).")
    ap.add_argument("--resume", type=str, default=None,
                     help="Explicit checkpoint .zip to resume from (e.g. a DIFFERENT run-name's "
                          "final checkpoint, for a hard-mining follow-up pass with a narrowed "
                          "terrain/slope/stair range) -- same --timesteps-as-target-total semantics "
                          "as --auto-resume. Takes precedence over --auto-resume if both given.")
    args = ap.parse_args()

    for curr_start, range_arg, name in [
        (args.terrain_curriculum_start, args.terrain_amplitude_range, "--terrain-amplitude-range"),
        (args.slope_curriculum_start, args.slope_deg_range, "--slope-deg-range"),
        (args.stair_curriculum_start, args.stair_height_range, "--stair-height-range"),
    ]:
        if curr_start is not None and range_arg is None:
            raise SystemExit(f"curriculum start given without {name}")

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
        terrain_amplitude_range=tuple(args.terrain_amplitude_range) if args.terrain_amplitude_range else None,
        slope_deg_range=tuple(args.slope_deg_range) if args.slope_deg_range else None,
        stair_height_range=tuple(args.stair_height_range) if args.stair_height_range else None,
        use_terrain_heightmap=args.use_terrain_heightmap,
        terrain_footstep_planning=args.terrain_footstep_planning,
        terrain_friction_cone=args.terrain_friction_cone,
        terrain_foothold_optimization=args.terrain_foothold_optimization,
        gait_style=args.gait_style,
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
    if args.terrain_curriculum_start is not None:
        env.set_attr("terrain_amp_max_current", args.terrain_curriculum_start)
    if args.slope_curriculum_start is not None:
        env.set_attr("slope_deg_max_current", args.slope_curriculum_start)
    if args.stair_curriculum_start is not None:
        env.set_attr("stair_height_max_current", args.stair_curriculum_start)
    env = VecMonitor(env)

    # --auto-resume's own lookup takes priority over --resume: on a crash-triggered retry
    # (run_resilient.sh always passes --auto-resume), THIS run's own latest checkpoint has
    # more progress than --resume's seed checkpoint and must win, or every retry would
    # restart the hard-mining pass from the curriculum stage's checkpoint instead of
    # continuing it.
    resume_ckpt = (latest_checkpoint(str(ckpt_dir)) if args.auto_resume else None) or args.resume
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
    callbacks = [checkpoint_callback, reward_logger]
    if args.terrain_curriculum_start is not None:
        callbacks.append(RampCallback("terrain_amp_max_current", "terrain_amp_max",
                                       args.terrain_curriculum_start, args.terrain_amplitude_range[1],
                                       args.terrain_curriculum_steps))
    if args.slope_curriculum_start is not None:
        callbacks.append(RampCallback("slope_deg_max_current", "slope_max_deg",
                                       args.slope_curriculum_start, args.slope_deg_range[1],
                                       args.slope_curriculum_steps))
    if args.stair_curriculum_start is not None:
        callbacks.append(RampCallback("stair_height_max_current", "stair_height_max",
                                       args.stair_curriculum_start, args.stair_height_range[1],
                                       args.stair_curriculum_steps))

    model.learn(total_timesteps=remaining, reset_num_timesteps=(resume_ckpt is None),
                callback=callbacks)

    model.save(str(ckpt_dir / "final"))
    env.save(str(ckpt_dir / "vecnormalize_final.pkl"))
    print(f"Done. Checkpoints in {ckpt_dir}, tensorboard logs in {log_dir}")


if __name__ == "__main__":
    main()
