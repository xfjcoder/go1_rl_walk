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
    ap.add_argument("--terrain-amplitude", type=float, nargs="+", default=None,
                    help="terrain amplitude(s) in m (peak-to-peak) to evaluate at; enables the rough-terrain env. "
                         "Default: 0 0.04 0.08 0.12 for a run trained on terrain, else flat ground.")
    ap.add_argument("--slope-deg", type=float, nargs="+", default=None,
                    help="slope angle(s) in deg to evaluate at (signed: + uphill, - downhill). "
                         "Default: 0 10 -10 20 -20 for a run trained with slopes, else flat.")
    ap.add_argument("--stair-height", type=float, nargs="+", default=None,
                    help="stair riser height(s) in m to evaluate at (signed: + ascending, - descending). "
                         "Default: 0 0.06 -0.06 0.12 -0.12 for a run trained with stairs, else flat.")
    ap.add_argument("--obstacle-height", type=float, nargs="+", default=None,
                    help="discrete obstacle height(s) in m to evaluate at. Default: 0 0.06 0.12 for a "
                         "run trained with obstacles, else flat.")
    ap.add_argument("--target-speed", type=float, nargs="+", default=None,
                    help="command speed(s) to evaluate at. Default: the run's fixed speed, or 0.3 0.5 0.8 1.0 "
                         "for a run trained with a command-speed range.")
    ap.add_argument("--robot-xml", default=None,
                    help="path to an alternate MJCF file (e.g. assets/go1_mesh.xml) to evaluate the checkpoint "
                         "against, in place of whatever assets/*.xml it was trained on. For sim-to-sim transfer "
                         "checks -- the checkpoint's observation/action space must still match exactly.")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    model_path = os.path.join(ckpt_dir, args.model) if args.model else latest_checkpoint(ckpt_dir)
    m = re.search(r"go1_flat_(\d+)_steps$", os.path.basename(model_path))
    vec = os.path.join(ckpt_dir, f"go1_flat_vecnormalize_{m.group(1)}_steps.pkl") if m \
        else os.path.join(ckpt_dir, "vecnormalize_final.pkl")
    with open(os.path.join(args.run_dir, "env_kwargs.json")) as f:
        env_kwargs = json.load(f)
    if args.robot_xml is not None:
        env_kwargs["xml_path"] = os.path.abspath(args.robot_xml)
    trained_on_terrain = bool(env_kwargs.get("terrain_amplitude_range"))
    trained_on_slope = bool(env_kwargs.get("slope_range"))
    trained_on_stairs = bool(env_kwargs.get("stair_height_range"))
    trained_on_obstacles = bool(env_kwargs.get("obstacle_height_range"))
    if args.target_speed is not None:
        speeds = args.target_speed
    elif trained_on_terrain:
        speeds = [0.3, 0.8]
    elif env_kwargs.get("command_speed_range"):
        speeds = [0.3, 0.5, 0.8, 1.0]
    else:
        speeds = [env_kwargs["target_speed"]]
    if args.terrain_amplitude is not None:
        amps = args.terrain_amplitude
    elif trained_on_terrain:
        amps = [0.0, 0.04, 0.08, 0.12]
    else:
        amps = [None]
    if args.slope_deg is not None:
        slopes = args.slope_deg
    elif trained_on_slope:
        slopes = [0.0, 10.0, -10.0, 20.0, -20.0]
    else:
        slopes = [None]
    if args.stair_height is not None:
        stairs = args.stair_height
    elif trained_on_stairs:
        stairs = [0.0, 0.06, -0.06, 0.12, -0.12]
    else:
        stairs = [None]
    if args.obstacle_height is not None:
        obstacles = args.obstacle_height
    elif trained_on_obstacles:
        obstacles = [0.0, 0.06, 0.12]
    else:
        obstacles = [None]
    print(f"model={model_path}.zip  kp={env_kwargs['kp']} kd={env_kwargs['kd']}")
    for a in amps:
        for s in slopes:
            for st in stairs:
                for ob in obstacles:
                    for v in speeds:
                        # fixed command, terrain amplitude, slope, stair height, obstacle height per evaluation
                        kw = dict(env_kwargs, target_speed=v, command_speed_range=None,
                                  terrain_amplitude_range=None, terrain_amplitude=a,
                                  slope_range=None, slope_deg=s,
                                  stair_height_range=None, stair_height=st,
                                  obstacle_height_range=None, obstacle_height=ob)
                        e = evaluate(model_path, vec, kw, args.episodes, args.seconds)
                        terr = "flat" if a is None else f"terrain {a * 100:.0f} cm"
                        slope_s = "" if s is None else f" slope {s:+.0f}deg"
                        stair_s = "" if st is None else f" stairs {st * 100:+.0f}cm"
                        obs_s = "" if ob is None else f" obstacles {ob * 100:.0f}cm"
                        print(f"{terr:>11s}{slope_s:>11s}{stair_s:>13s}{obs_s:>14s} | command {v:.2f} m/s: fall_rate={e['fall_rate']:.2f}  survival={e['mean_survival_s']:5.1f}s/{args.seconds:.0f}s"
                              f"  speed mean={e['speed_mean']:+.3f} median={e['speed_median']:+.3f}"
                              f"  |y|={e['abs_y_mean']:.2f} m  |yaw|={e['abs_yaw_mean']:.1f} deg", flush=True)


if __name__ == "__main__":
    main()
