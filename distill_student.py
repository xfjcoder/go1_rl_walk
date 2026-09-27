"""
Distill runs/go2_latency_teacher (privileged per-episode latency observation, 53-dim) into a
realistic "student" policy that only has the ordinary 51-dim observation (no ground-truth
latency signal) -- the deployment-relevant half of teacher/student distillation for
sim-to-real latency robustness. See README's "Teacher/student latency distillation" section
for the full context and results.

Two stages:
  1. Truncate: build the student's STARTING POINT by dropping the teacher's 2 privileged
     input columns from its first layer, copying every other weight unchanged (the exact
     reverse operation of expand_obs_checkpoint.py's zero-padding). Verified: with the
     privileged inputs forced to 0 (== what "no latency" always looked like to the teacher),
     the truncated student's action is bit-identical to the teacher's -- the ONLY behavior
     removed is whatever depended on those 2 now-absent inputs. This gives the student a
     head start on everything else (walking, gait, terrain handling) instead of starting
     from random weights.
  2. Distill: collect (student_observation, teacher_action) pairs by rolling out the TEACHER
     (with real, randomly varying per-episode latency, matching its own training
     distribution) and train the student's policy network via supervised regression (MSE)
     to reproduce the teacher's action using ONLY the student's own, non-privileged
     observation at that same instant.

CAVEAT, stated up front, not discovered after the fact: the student's observation is a
single, memoryless timestep -- no history buffer. Standard approaches for exactly this
problem (e.g. Rapid Motor Adaptation / RMA-style adaptation modules) use a SHORT HISTORY of
past observations as the student's input specifically because a single instantaneous
proprioceptive reading carries very little information about processing delay by itself --
delay is a property of a *sequence*, not a snapshot. This project's env has no
history-buffering mechanism at all, so this script tests the simpler, single-step version
directly rather than assuming it will (or won't) work -- if it doesn't recover much of the
teacher's specialization, that's itself a real, informative result about what's actually
needed (a history-based adaptation module), not a bug in this script.

Usage:
    python distill_student.py --teacher runs/go2_latency_teacher --out-dir runs/go2_latency_student
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv

PRIVILEGED_DIMS = 2  # the 2 extra dims privileged_latency_obs appends (see envs/go1_env.py)


def make_env(seed, env_kwargs):
    def _init():
        e = Go1FlatEnv(render_mode=None, domain_randomize=True, **env_kwargs)
        e.reset(seed=seed)
        return e
    return _init


def truncate_student(teacher_model, teacher_vecnorm, student_env, old_obs_dim):
    """Stage 1: build the student's PPO model by dropping the teacher's last PRIVILEGED_DIMS
    input columns from its first layer, copying every other weight unchanged."""
    new_obs_dim = old_obs_dim - PRIVILEGED_DIMS
    student = PPO("MlpPolicy", student_env,
                   policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
                   device="cpu")
    old_sd = teacher_model.policy.state_dict()
    new_sd = student.policy.state_dict()
    for k, v in new_sd.items():
        old_w = old_sd[k]
        if old_w.shape == v.shape:
            new_sd[k] = old_w.clone()
        elif old_w.dim() == 2 and old_w.shape[1] == old_obs_dim and v.shape[1] == new_obs_dim \
                and old_w.shape[0] == v.shape[0]:
            new_sd[k] = old_w[:, :new_obs_dim].clone()
            print(f"  truncated {k}: {tuple(old_w.shape)} -> {tuple(v.shape)} (dropped last "
                  f"{PRIVILEGED_DIMS} input columns)")
        else:
            raise ValueError(f"unexpected shape mismatch at {k}: old {tuple(old_w.shape)} vs new {tuple(v.shape)}")
    student.policy.load_state_dict(new_sd)

    student_vecnorm = VecNormalize(student_env, norm_obs=True, norm_reward=teacher_vecnorm.norm_reward,
                                    clip_obs=teacher_vecnorm.clip_obs, clip_reward=teacher_vecnorm.clip_reward,
                                    gamma=teacher_vecnorm.gamma)
    student_vecnorm.obs_rms.mean = teacher_vecnorm.obs_rms.mean[:new_obs_dim].copy()
    student_vecnorm.obs_rms.var = teacher_vecnorm.obs_rms.var[:new_obs_dim].copy()
    student_vecnorm.obs_rms.count = teacher_vecnorm.obs_rms.count
    student_vecnorm.ret_rms = teacher_vecnorm.ret_rms
    student_vecnorm.training = False
    return student, student_vecnorm


def verify_truncation(teacher_model, teacher_vecnorm_path, student_model, teacher_kwargs):
    """Truncation drops the teacher's last PRIVILEGED_DIMS input columns from its first
    layer -- this is exact IFF those 2 NETWORK INPUTS (i.e. post-VecNormalize-normalization)
    are held at 0, since a dropped column contributes weight*0 = 0 either way. Critically,
    this is NOT the same as the RAW privileged latency being 0: VecNormalize centers on the
    TRAINING distribution's mean (e.g. ~2, from a Uniform(0,4) latency range), so raw
    latency=0 normalizes to a nonzero value (observed: ~-1.05), not 0 -- an earlier version
    of this check sampled raw latency=0 via the environment and wrongly expected that to
    correspond to network-input=0, which is why it failed even though the weight copy itself
    is exact by construction (verified here more directly instead: take a REAL observation
    from a normal episode, at ANY latency, and manually zero only the final 2 already-
    normalized network inputs before comparing)."""
    env = SubprocVecEnv([make_env(0, teacher_kwargs)])
    vn = VecNormalize.load(teacher_vecnorm_path, env)
    vn.training = False
    obs = vn.reset()
    for _ in range(20):  # walk a few steps so it's not just the reset pose
        action, _ = teacher_model.predict(obs, deterministic=True)
        obs, _, _, _ = vn.step(action)
    obs_zeroed = obs.copy()
    obs_zeroed[:, -PRIVILEGED_DIMS:] = 0.0
    a_t, _ = teacher_model.predict(obs_zeroed, deterministic=True)
    a_s, _ = student_model.predict(obs_zeroed[:, :-PRIVILEGED_DIMS], deterministic=True)
    action_match = np.allclose(a_t, a_s, atol=1e-4)
    print(f"truncation check: action {'PASS' if action_match else 'FAIL'} "
          f"(max diff {np.abs(a_t - a_s).max():.6f}, at a real mid-episode observation with the "
          f"2 privileged network inputs held at exactly 0)")
    vn.close()
    if not action_match:
        raise SystemExit("Truncation did not reproduce the teacher's behavior at zero privileged "
                          "network input -- do not use it.")


def collect_dataset(teacher_model, teacher_vecnorm_path, teacher_kwargs, n_envs, seconds, rounds, seed0=20_000):
    """Roll out the TEACHER (driving the environment, with its own real per-episode latency)
    across `rounds` batches of `n_envs` parallel episodes, recording the student's own
    (non-privileged) observation and the teacher's action at every step."""
    obs_buf, act_buf = [], []
    max_steps = int(seconds * 50)
    for r in range(rounds):
        env = SubprocVecEnv([make_env(seed0 + r * n_envs + i, teacher_kwargs) for i in range(n_envs)])
        env = VecNormalize.load(teacher_vecnorm_path, env)
        env.training = False
        obs = env.reset()
        for _ in range(max_steps):
            action, _ = teacher_model.predict(obs, deterministic=True)
            obs_buf.append(obs[:, :-PRIVILEGED_DIMS].copy())
            act_buf.append(action.copy())
            obs, _, dones, _ = env.step(action)
        env.close()
        print(f"  round {r + 1}/{rounds} done ({(r + 1) * n_envs * max_steps} transitions so far)")
    X = np.concatenate(obs_buf, axis=0).astype(np.float32)
    Y = np.concatenate(act_buf, axis=0).astype(np.float32)
    return X, Y


def train_supervised(student_model, X, Y, epochs, batch_size, lr):
    """Supervised MSE regression of the student's policy network onto the teacher's actions.
    Only the policy (actor) network is trained -- the value network is never used at
    inference time (play.py/eval_policy.py/gait_stats.py only ever call .predict())."""
    device = "cpu"
    policy = student_model.policy
    policy.train()
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    X_t = torch.as_tensor(X, device=device)
    Y_t = torch.as_tensor(Y, device=device)
    n = X_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], Y_t[idx]
            latent_pi = policy.mlp_extractor.forward_actor(xb)
            pred_mean = policy.action_net(latent_pi)
            loss = nn.functional.mse_loss(pred_mean, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch + 1}/{epochs}: mean MSE loss = {total_loss / n_batches:.6f}")
    policy.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="--run-dir of the teacher (e.g. runs/go2_latency_teacher)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rollout-envs", type=int, default=16)
    ap.add_argument("--rollout-seconds", type=float, default=20.0)
    ap.add_argument("--rollout-rounds", type=int, default=20,
                     help="rollout-envs * (rollout-seconds*50) transitions per round")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    teacher_model_path = os.path.join(args.teacher, "checkpoints", "go1_flat_final")
    teacher_vecnorm_path = os.path.join(args.teacher, "checkpoints", "vecnormalize_final.pkl")
    with open(os.path.join(args.teacher, "env_kwargs.json")) as f:
        teacher_kwargs = json.load(f)
    assert teacher_kwargs.get("privileged_latency_obs"), "--teacher's env_kwargs.json must have privileged_latency_obs=true"

    student_kwargs = dict(teacher_kwargs, privileged_latency_obs=False)
    # student never gets to see the ground-truth latency, but DOES experience its real effect
    # on the environment -- action_latency_range/observation_latency_range stay set.

    teacher_model = PPO.load(teacher_model_path, device="cpu")
    old_obs_dim = teacher_model.observation_space.shape[0]

    student_env = SubprocVecEnv([make_env(0, student_kwargs)])
    teacher_vn_for_stats = VecNormalize.load(teacher_vecnorm_path, SubprocVecEnv([make_env(0, teacher_kwargs)]))

    print("Stage 1: truncating the teacher's privileged input columns...")
    student_model, student_vecnorm = truncate_student(teacher_model, teacher_vn_for_stats, student_env, old_obs_dim)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, "go1_flat_final")
    vecnorm_path = os.path.join(ckpt_dir, "vecnormalize_final.pkl")
    student_model.num_timesteps = teacher_model.num_timesteps
    student_model.save(model_path)
    student_vecnorm.save(vecnorm_path)
    with open(os.path.join(args.out_dir, "env_kwargs.json"), "w") as f:
        json.dump(student_kwargs, f, indent=2)

    print("Verifying the truncation reproduces the teacher's behavior at zero privileged network input...")
    verify_truncation(teacher_model, teacher_vecnorm_path, student_model, teacher_kwargs)

    print(f"Stage 2: collecting a distillation dataset from {args.rollout_rounds} rounds of "
          f"{args.rollout_envs} parallel teacher rollouts ({args.rollout_seconds}s each)...")
    X, Y = collect_dataset(teacher_model, teacher_vecnorm_path, teacher_kwargs,
                            args.rollout_envs, args.rollout_seconds, args.rollout_rounds)
    print(f"  collected {X.shape[0]} transitions, obs_dim={X.shape[1]}, action_dim={Y.shape[1]}")

    print(f"Stage 3: supervised distillation ({args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr})...")
    train_supervised(student_model, X, Y, args.epochs, args.batch_size, args.lr)

    student_model.save(model_path)
    print(f"Saved distilled student to {model_path}.zip, usable directly as --run-dir {args.out_dir}")


if __name__ == "__main__":
    main()
