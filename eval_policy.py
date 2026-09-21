"""
Evaluate a trained policy over many randomized episodes (same reset randomization as training:
joint/height/yaw jitter + floor friction). Reports fall rate, forward speed and drift so runs and
checkpoints can be compared on the same footing -- single episodes are misleading.

Usage:
    python eval_policy.py --run-dir runs/pilot_a                       # latest checkpoint of the run
    python eval_policy.py --run-dir runs/pilot_a --model go1_flat_2000000_steps
    python eval_policy.py --run-dir runs/pilot_a --episodes 32 --seconds 20 --target-speed 0.3
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv


def latest_checkpoint(ckpt_dir):
    """Newest go1_flat_<steps>_steps by step count (falls back to go1_flat_final)."""
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, "go1_flat_*_steps.zip")):
        m = re.search(r"go1_flat_(\d+)_steps\.zip$", f)
        if m:
            steps.append((int(m.group(1)), f))
    if steps:
        return max(steps)[1][:-4]
    return os.path.join(ckpt_dir, "go1_flat_final")


def make_env(seed, env_kwargs):
    def _init():
        env = Go1FlatEnv(render_mode=None, domain_randomize=True, **env_kwargs)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model_path, vecnorm_path, env_kwargs, episodes=16, seconds=20.0, seed0=10_000):
    env = SubprocVecEnv([make_env(seed0 + i, env_kwargs) for i in range(episodes)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training, env.norm_reward = False, False
    model = PPO.load(model_path, device="cpu")
    obs = env.reset()
    max_steps = int(seconds * 50)
    done_ep = np.zeros(episodes, dtype=bool)
    res = [None] * episodes
    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        for i in range(episodes):
            if not done_ep[i] and (dones[i] or step == max_steps):
                pos = infos[i]["base_pos"]
                fell = bool(dones[i]) and not infos[i].get("TimeLimit.truncated", False)
                res[i] = dict(fell=fell, t=step / 50.0, x=float(pos[0]), y=float(pos[1]),
                              yaw=float(np.degrees(infos[i]["yaw"])))
                done_ep[i] = True
        if done_ep.all():
            break
    env.close()
    r = res
    fell = np.array([x["fell"] for x in r])
    t = np.array([x["t"] for x in r])
    speed = np.array([x["x"] / x["t"] for x in r])
    return dict(
        episodes=episodes, fall_rate=float(fell.mean()), mean_survival_s=float(t.mean()),
        speed_mean=float(speed.mean()), speed_median=float(np.median(speed)),
        speed_survivors=float(speed[~fell].mean()) if (~fell).any() else 0.0,
        abs_y_mean=float(np.mean([abs(x["y"]) for x in r])), abs_yaw_mean=float(np.mean([abs(x["yaw"]) for x in r])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default=None, help="checkpoint name inside <run-dir>/checkpoints (no .zip)")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--target-speed", type=float, nargs="+", default=None,
                    help="command speed(s) to evaluate at. Default: the run's fixed speed, or 0.3 0.5 0.8 1.0 "
                         "for a run trained with a command-speed range.")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    model_path = os.path.join(ckpt_dir, args.model) if args.model else latest_checkpoint(ckpt_dir)
    m = re.search(r"go1_flat_(\d+)_steps$", os.path.basename(model_path))
    vec = os.path.join(ckpt_dir, f"go1_flat_vecnormalize_{m.group(1)}_steps.pkl") if m \
        else os.path.join(ckpt_dir, "vecnormalize_final.pkl")
    with open(os.path.join(args.run_dir, "env_kwargs.json")) as f:
        env_kwargs = json.load(f)
    if args.target_speed is not None:
        speeds = args.target_speed
    elif env_kwargs.get("command_speed_range"):
        speeds = [0.3, 0.5, 0.8, 1.0]
    else:
        speeds = [env_kwargs["target_speed"]]
    print(f"model={model_path}.zip  kp={env_kwargs['kp']} kd={env_kwargs['kd']}")
    for v in speeds:
        kw = dict(env_kwargs, target_speed=v, command_speed_range=None)   # fixed command per evaluation
        e = evaluate(model_path, vec, kw, args.episodes, args.seconds)
        print(f"command {v:.2f} m/s: episodes={e['episodes']}  fall_rate={e['fall_rate']:.2f}  survival={e['mean_survival_s']:.1f}s/{args.seconds:.0f}s"
              f"  speed mean={e['speed_mean']:+.3f} median={e['speed_median']:+.3f}"
              f"  |y|={e['abs_y_mean']:.2f} m  |yaw|={e['abs_yaw_mean']:.1f} deg")


if __name__ == "__main__":
    main()
