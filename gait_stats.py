"""
Per-foot gait diagnostics for a trained policy (deterministic), to spot lopsided gaits that
fall rate / speed alone hide: step rate, swing/stance time, swing height and stance travel per foot.

Usage:
    python gait_stats.py --run-dir runs/e_long_d                       # newest checkpoint
    python gait_stats.py --run-dir runs/e_long_d --model go1_flat_5000000_steps --seed 2
"""
import argparse
import json
import os
import re

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv
from eval_policy import latest_checkpoint

NAMES = ["FR", "FL", "RR", "RL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--terrain-amplitude", type=float, default=None, help="evaluate on rough terrain of this amplitude (m)")
    ap.add_argument("--target-speed", type=float, default=None, help="fixed command speed (default: the run's)")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    ckpt = os.path.join(args.run_dir, "checkpoints")
    path = os.path.join(ckpt, args.model) if args.model else latest_checkpoint(ckpt)
    m = re.search(r"go1_flat_(\d+)_steps$", os.path.basename(path))
    vec = os.path.join(ckpt, f"go1_flat_vecnormalize_{m.group(1)}_steps.pkl") if m else os.path.join(ckpt, "vecnormalize_final.pkl")
    kw = json.load(open(os.path.join(args.run_dir, "env_kwargs.json")))
    if args.target_speed is not None:
        kw["target_speed"] = args.target_speed
    kw["command_speed_range"] = None      # fixed command for diagnostics
    kw["terrain_amplitude_range"] = None
    kw["terrain_amplitude"] = args.terrain_amplitude
    model = PPO.load(path, device="cpu")
    env = DummyVecEnv([lambda: Go1FlatEnv(render_mode=None, domain_randomize=True, **kw)])
    env.seed(args.seed)
    env = VecNormalize.load(vec, env)
    env.training, env.norm_reward = False, False
    e = env.envs[0]
    obs = env.reset()
    touch, fx, fz, ys, yaws, vb = [], [], [], [], [], []
    n_steps = int(args.seconds * 50)
    for _ in range(n_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(act)
        if done[0]:
            break
        touch.append(e._foot_contacts())
        fx.append(e.data.site_xpos[e._foot_site_ids, 0] - e.data.qpos[0])
        fz.append(e.data.site_xpos[e._foot_site_ids, 2] - e._foot_radius)
        ys.append(info[0]["base_pos"][1]); yaws.append(np.degrees(info[0]["yaw"]))
        vb.append(e._quat_rotate_inv(e.data.sensordata[e._imu_quat_adr:e._imu_quat_adr + 4],
                                     e.data.sensordata[e._imu_vel_adr:e._imu_vel_adr + 3]))
    T, fx, fz, vb = np.array(touch), np.array(fx), np.array(fz), np.array(vb)
    n = len(T)
    print(f"command={kw['target_speed']:.2f} m/s  {path}.zip  seed={args.seed}  steps={n}  x={info[0]['base_pos'][0]:+.2f} y={ys[-1]:+.2f} yaw={yaws[-1]:+.1f}deg  "
          f"body vx={vb[:, 0].mean():+.3f} vy={vb[:, 1].mean():+.3f}")
    print("contact timeline (1 char = 0.02 s, # = on ground), t=10..11.5 s:")
    for k in range(4):
        print(f"  {NAMES[k]} " + "".join("#" if c else "." for c in T[500:575, k]))
    s0 = min(100, n // 4)
    print(f"per-foot over t>={s0 / 50:.0f}s:  steps/s  duty  swing_ms  stance_ms  stance_travel_cm  swing_peak_cm")
    rates = []
    for k in range(4):
        x = T[s0:, k]
        ch = np.flatnonzero(np.diff(x.astype(int))) + 1
        segs = np.split(np.arange(len(x)), ch)
        sw = [len(s) * 0.02 for s in segs if not x[s[0]]]
        st = [len(s) * 0.02 for s in segs if x[s[0]]]
        trav = [abs(fx[s0:][s[-1], k] - fx[s0:][s[0], k]) for s in segs if x[s[0]] and len(s) > 3]
        pk = [fz[s0:][s, k].max() for s in segs if not x[s[0]] and len(s) > 1]
        td = ((x[1:]) & (~x[:-1])).sum()
        rates.append(td / (len(x) / 50))
        print(f"  {NAMES[k]}:  {rates[-1]:5.2f}  {x.mean():5.2f}  {np.mean(sw) * 1000:7.0f}  {np.mean(st) * 1000:8.0f}  "
              f"{np.mean(trav) * 100 if trav else 0:14.1f}  {np.mean(pk) * 100 if pk else 0:12.1f}")
    nc = T.sum(1)
    print(f"step-rate spread (max/min) = {max(rates) / max(min(rates), 1e-6):.2f}   "
          f"feet-down: " + " ".join(f"{k}:{(nc == k).mean():.2f}" for k in range(5)))


if __name__ == "__main__":
    main()
