"""
Stage 10a: randomized multi-episode evaluation of point-to-goal navigation (see navigate.py for
the controller itself and the design rationale). Mirrors eval_policy.py's own pattern -- a single
demo run proves nothing, judge this the same randomized-multi-seed way every other stage in this
project is judged (see memory: go1-eval-method).

Each episode samples a goal at a random bearing (0-360 deg, so backward/lateral-only goals are
included) and distance within --goal-distance-range, optionally on rough terrain within
--terrain-amplitude. Reports success rate, time-to-goal (successes only), and path efficiency
(straight-line distance / actual distance traveled -- 1.0 is perfectly direct).

Usage:
    python eval_navigate.py --episodes 32
    python eval_navigate.py --episodes 32 --terrain-amplitude 0.0 0.08
"""
import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv
from eval_policy import latest_checkpoint


def make_env(seed, env_kwargs):
    def _init():
        env = Go1FlatEnv(render_mode=None, domain_randomize=True, **env_kwargs)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model_path, vecnorm_path, env_kwargs, episodes, seconds, kp, tolerance,
             goal_dist_range, speed_lo, speed_hi, lat_lo, lat_hi, seed0=20_000):
    # command_speed_range/lateral_speed_range must already be None and target_speed/
    # target_lateral_speed already 0.0 in env_kwargs (set by main(), mirroring navigate.py) --
    # otherwise reset()'s own random resample would bake a stale command into the very first
    # returned observation, one step before our own override below ever takes effect.
    assert env_kwargs.get("command_speed_range") is None and env_kwargs.get("target_speed") == 0.0
    env = SubprocVecEnv([make_env(seed0 + i, env_kwargs) for i in range(episodes)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training, env.norm_reward = False, False
    model = PPO.load(model_path, device="cpu")

    rng = np.random.default_rng(seed0)
    bearing = rng.uniform(0.0, 2 * np.pi, episodes)
    dist0 = rng.uniform(goal_dist_range[0], goal_dist_range[1], episodes)
    goal_x, goal_y = dist0 * np.cos(bearing), dist0 * np.sin(bearing)

    obs = env.reset()
    # One null-command step per episode to read back the true start pose (reset() itself returns
    # no info; domain_randomize=True jitters yaw +-0.1 rad at reset, so don't assume yaw=0 -- 20ms
    # of standing-still motion is negligible against goal distances of order metres).
    env.set_attr("target_speed", 0.0)
    env.set_attr("target_lateral_speed", 0.0)
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, infos = env.step(action)
    x = np.array([i["base_pos"][0] for i in infos])
    y = np.array([i["base_pos"][1] for i in infos])
    yaw = np.array([i["yaw"] for i in infos])
    path_len = np.zeros(episodes)

    max_steps = int(seconds * 50)
    done_ep = np.zeros(episodes, dtype=bool)
    res = [None] * episodes
    for step in range(2, max_steps + 1):
        dx, dy = goal_x - x, goal_y - y
        dist = np.hypot(dx, dy)
        reached_now = dist <= tolerance
        c, s = np.cos(yaw), np.sin(yaw)
        fwd_err, lat_err = dx * c + dy * s, -dx * s + dy * c
        fwd_cmd = np.clip(kp * fwd_err, speed_lo, speed_hi)
        lat_cmd = np.clip(kp * lat_err, lat_lo, lat_hi)
        fwd_cmd = np.where(reached_now, 0.0, fwd_cmd)
        lat_cmd = np.where(reached_now, 0.0, lat_cmd)
        for i in range(episodes):
            if not done_ep[i]:
                env.env_method("set_command", float(fwd_cmd[i]), float(lat_cmd[i]), indices=[i])

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        new_x = np.array([inf["base_pos"][0] for inf in infos])
        new_y = np.array([inf["base_pos"][1] for inf in infos])
        path_len += np.hypot(new_x - x, new_y - y)
        x, y = new_x, new_y
        yaw = np.array([inf["yaw"] for inf in infos])

        for i in range(episodes):
            if done_ep[i]:
                continue
            if reached_now[i] and res[i] is None:
                res[i] = dict(reached=True, t=step / 50.0)
            if dones[i] or step == max_steps:
                fell = bool(dones[i]) and not infos[i].get("TimeLimit.truncated", False)
                final_dist = float(np.hypot(goal_x[i] - x[i], goal_y[i] - y[i]))
                straight = float(np.hypot(goal_x[i], goal_y[i]))
                base = res[i] or dict(reached=False, t=step / 50.0)
                res[i] = dict(reached=base["reached"], t=base["t"], fell=fell,
                               final_dist=final_dist, path_len=float(path_len[i]), straight=straight)
                done_ep[i] = True
        if done_ep.all():
            break
    env.close()

    reached = np.array([r["reached"] for r in res])
    fell = np.array([r["fell"] for r in res])
    t_reach = np.array([r["t"] for r in res if r["reached"]])
    eff = np.array([r["straight"] / r["path_len"] for r in res if r["reached"] and r["path_len"] > 1e-6])
    return dict(
        episodes=episodes, success_rate=float(reached.mean()), fall_rate=float(fell.mean()),
        time_to_goal_mean=float(t_reach.mean()) if t_reach.size else float("nan"),
        path_efficiency_mean=float(eff.mean()) if eff.size else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="pretrained/go2_latback")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--goal-distance-range", type=float, nargs=2, default=[1.5, 4.0])
    ap.add_argument("--terrain-amplitude", type=float, nargs="+", default=[None],
                    help="terrain amplitude(s) in m (peak-to-peak) to evaluate at. Default: flat "
                         "ground only (Stage 10a's scope -- slopes/stairs not attempted here).")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    model_path = os.path.join(ckpt_dir, args.model) if args.model else latest_checkpoint(ckpt_dir)
    vec = os.path.join(ckpt_dir, "vecnormalize_final.pkl")
    with open(os.path.join(args.run_dir, "env_kwargs.json")) as f:
        env_kwargs = json.load(f)

    # Capture the checkpoint's own trained command range for clipping BEFORE blanking it out below
    # (same pattern as navigate.py: command_speed_range/lateral_speed_range -> None means "don't
    # resample every reset", not "no limit" -- the actual limits are these saved values).
    speed_lo, speed_hi = env_kwargs.get("command_speed_range") or (env_kwargs["target_speed"],) * 2
    lat_lo, lat_hi = env_kwargs.get("lateral_speed_range") or (env_kwargs.get("target_lateral_speed", 0.0),) * 2

    print(f"model={model_path}.zip  goal distance {args.goal_distance_range[0]}-{args.goal_distance_range[1]} m, "
          f"bearing 0-360deg, kp={args.kp}, tolerance={args.tolerance} m")
    for amp in args.terrain_amplitude:
        kw = dict(env_kwargs, max_episode_seconds=args.seconds,   # not saved in env_kwargs.json;
                  command_speed_range=None, lateral_speed_range=None,
                  target_speed=0.0, target_lateral_speed=0.0,
                  terrain_amplitude_range=None, terrain_amplitude=amp,
                  slope_range=None, slope_deg=None, stair_height_range=None, stair_height=None,
                  obstacle_height_range=None, obstacle_height=None)
        e = evaluate(model_path, vec, kw, args.episodes, args.seconds, args.kp, args.tolerance,
                     args.goal_distance_range, speed_lo, speed_hi, lat_lo, lat_hi)
        terr = "flat" if amp is None else f"terrain {amp * 100:.0f} cm"
        print(f"{terr:>13s}: success_rate={e['success_rate']:.2f}  fall_rate={e['fall_rate']:.2f}  "
              f"time_to_goal={e['time_to_goal_mean']:5.1f}s  path_efficiency={e['path_efficiency_mean']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
