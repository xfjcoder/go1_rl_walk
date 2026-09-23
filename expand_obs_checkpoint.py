"""
Warm-start a policy with a LARGER observation space from an existing checkpoint, by copying every
weight except the very first Linear layer of the policy/value MLPs (whose input dimension must
grow to match). The new input columns are zero-initialized, so the expanded model is mathematically
IDENTICAL to the original at t=0 regardless of what the new observation channels actually contain --
fine-tuning then discovers how (and whether) to use them. This avoids retraining an entire staged
stack (h_clock -> ... -> p_stairs, dozens of millions of steps) from scratch just to add one signal.

Built for --use-terrain-heightmap (51-dim blind observation -> 60-dim with a local heightmap), but
works for any observation-space growth where the new dims are appended at the end and every other
layer's shape is unaffected.

Usage:
    python expand_obs_checkpoint.py \
        --model pretrained/p_stairs/checkpoints/go1_flat_final \
        --vecnormalize pretrained/p_stairs/checkpoints/vecnormalize_final.pkl \
        --env-kwargs pretrained/p_stairs/env_kwargs.json \
        --new-env-kwarg use_terrain_heightmap=true \
        --out-dir runs/warmstart_heightmap

Verifies itself before trusting the result: seeds a copy of the old (small-obs) and new (large-obs)
envs identically so they reach the same physical state, then checks the new observation's first
old_obs_dim entries exactly match the old observation, and the expanded model's deterministic action
exactly matches the original's. Refuses to exit cleanly if either check fails.
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv

NEW_DIM_VAR_INIT = 0.01     # (0.1 m std)^2 -- a reasonable guess for local heightmap deltas (metres)
NEW_RMS_COUNT = 10_000.0    # deliberately much smaller than a converged checkpoint's count (~1e8),
                            # so the new dims' running mean/var can adapt to their true distribution
                            # within a few M steps of fine-tuning instead of being stuck near this
                            # initial guess. VecNormalize shares one count across all dims, so this
                            # is a compromise: too small risks a noisy first update swinging the
                            # OLD dims' already-good stats; too large (matching the real count)
                            # would leave the NEW dims' stats frozen near this guess indefinitely.


def parse_kwarg(s: str):
    k, v = s.split("=", 1)
    if v.lower() in ("true", "false"):
        return k, v.lower() == "true"
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the existing checkpoint (no .zip)")
    ap.add_argument("--vecnormalize", required=True)
    ap.add_argument("--env-kwargs", required=True, help="env_kwargs.json from the existing run")
    ap.add_argument("--new-env-kwarg", action="append", default=[],
                     help="key=value to add/override in env_kwargs for the new (larger-obs) env, "
                          "e.g. --new-env-kwarg use_terrain_heightmap=true. Repeatable.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.env_kwargs) as f:
        env_kwargs = json.load(f)
    old_only_kwargs = dict(env_kwargs)   # exactly what the OLD checkpoint's env was built with
    for kv in args.new_env_kwarg:
        k, v = parse_kwarg(kv)
        env_kwargs[k] = v

    old_model = PPO.load(args.model, device="cpu")
    old_obs_dim = old_model.observation_space.shape[0]

    new_env = DummyVecEnv([lambda: Go1FlatEnv(render_mode=None, domain_randomize=True, **env_kwargs)])
    new_obs_dim = new_env.observation_space.shape[0]
    added = new_obs_dim - old_obs_dim
    if added <= 0:
        raise SystemExit(f"new obs_dim ({new_obs_dim}) must be larger than the old one ({old_obs_dim})")
    print(f"expanding observation: {old_obs_dim} -> {new_obs_dim} dims (+{added})")

    new_model = PPO("MlpPolicy", new_env, policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
                     device="cpu")

    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    for k, v in new_sd.items():
        if k not in old_sd:
            raise KeyError(f"key {k} present in the new (larger) policy but not the old one -- "
                            f"architectures other than the observation size must match exactly")
        old_w = old_sd[k]
        if old_w.shape == v.shape:
            new_sd[k] = old_w.clone()
        elif old_w.dim() == 2 and old_w.shape[1] == old_obs_dim and v.shape[1] == new_obs_dim \
                and old_w.shape[0] == v.shape[0]:
            expanded = torch.zeros_like(v)
            expanded[:, :old_obs_dim] = old_w
            new_sd[k] = expanded
            print(f"  expanded {k}: {tuple(old_w.shape)} -> {tuple(v.shape)} (new columns zero-init)")
        else:
            raise ValueError(f"unexpected shape mismatch at {k}: old {tuple(old_w.shape)} vs new {tuple(v.shape)} "
                              f"-- this script only handles growth in the observation (input) dimension")
    new_model.policy.load_state_dict(new_sd)

    old_vecnorm = VecNormalize.load(args.vecnormalize, DummyVecEnv([lambda: Go1FlatEnv(render_mode=None, **old_only_kwargs)]))
    new_vecnorm = VecNormalize(new_env, norm_obs=True, norm_reward=old_vecnorm.norm_reward,
                                clip_obs=old_vecnorm.clip_obs, clip_reward=old_vecnorm.clip_reward,
                                gamma=old_vecnorm.gamma)
    new_vecnorm.obs_rms.mean = np.concatenate([old_vecnorm.obs_rms.mean, np.zeros(added, dtype=np.float64)])
    new_vecnorm.obs_rms.var = np.concatenate([old_vecnorm.obs_rms.var, np.full(added, NEW_DIM_VAR_INIT, dtype=np.float64)])
    new_vecnorm.obs_rms.count = NEW_RMS_COUNT
    new_vecnorm.ret_rms = old_vecnorm.ret_rms  # reward normalization unaffected by obs dim change
    new_vecnorm.training = True

    # Match train.py's own naming convention exactly (checkpoints/go1_flat_final.zip +
    # checkpoints/vecnormalize_final.pkl), so `--resume <out-dir>/checkpoints/go1_flat_final.zip`
    # finds this VecNormalize automatically via vecnormalize_path_for() instead of silently
    # discarding it and starting fresh -- and so <out-dir> is directly usable as a --run-dir for
    # eval_policy.py/gait_stats.py/play.py even before any fine-tuning.
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, "go1_flat_final")
    vecnorm_path = os.path.join(ckpt_dir, "vecnormalize_final.pkl")
    new_model.save(model_path)
    new_vecnorm.save(vecnorm_path)
    with open(os.path.join(args.out_dir, "env_kwargs.json"), "w") as f:
        json.dump(env_kwargs, f, indent=2)
    print(f"saved {model_path}.zip, {vecnorm_path}, and env_kwargs.json -- "
          f"usable directly as --run-dir {args.out_dir}, or --resume {model_path}.zip")

    # Sanity check: seed the old (small-obs) and new (large-obs) envs identically, so they reach the
    # SAME underlying physical state on reset -- then confirm (a) the new obs really is the old obs
    # with the new channels appended, and (b) with those new channels not artificially forced to 0
    # (whatever the real terrain gives, e.g. a slope/stairs episode), the expanded model's
    # deterministic action still exactly matches the original's, since the new input columns are
    # zero-weighted regardless of what the new channels actually contain. If this fails, DO NOT
    # trust the warm-start.
    old_env = DummyVecEnv([lambda: Go1FlatEnv(render_mode=None, domain_randomize=True, **old_only_kwargs)])
    old_vn = VecNormalize.load(args.vecnormalize, old_env)
    old_vn.training = False
    fresh_new_env = DummyVecEnv([lambda: Go1FlatEnv(render_mode=None, domain_randomize=True, **env_kwargs)])
    new_vn = VecNormalize.load(vecnorm_path, fresh_new_env)
    new_vn.training = False
    old_env.seed(0)
    fresh_new_env.seed(0)
    o_old = old_vn.reset()
    o_new = new_vn.reset()
    obs_match = np.allclose(o_old, o_new[:, :old_obs_dim], atol=1e-5)
    print(f"matched-seed observation check: {'PASS' if obs_match else 'FAIL'} "
          f"(new obs[:{old_obs_dim}] should equal the old obs exactly)")
    a_old, _ = old_model.predict(o_old, deterministic=True)
    a_new, _ = new_model.predict(o_new, deterministic=True)
    action_match = np.allclose(a_old, a_new, atol=1e-4)
    print(f"warm-start behavioral check (old vs expanded action on the same physical state): "
          f"{'PASS -- identical' if action_match else 'FAIL -- see below'} "
          f"(max diff {np.abs(a_old - a_new).max():.6f})")
    if not (obs_match and action_match):
        raise SystemExit("Warm-start did not reproduce the original policy's behavior -- do not use it.")


if __name__ == "__main__":
    main()
