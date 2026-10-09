"""
Stage 10a/10b: randomized multi-episode evaluation of point-to-goal navigation, optionally with
static obstacles to avoid (see navigate.py for the controller itself and the design rationale).
Mirrors eval_policy.py's own pattern -- a single demo run proves nothing, judge this the same
randomized-multi-seed way every other stage in this project is judged (see memory: go1-eval-method).

Each episode samples a goal at a random bearing (0-360 deg, so backward/lateral-only goals are
included) and distance within --goal-distance-range, optionally on rough terrain within
--terrain-amplitude and/or with --nav-obstacles static obstacles scattered in the course (each
episode's own independently-randomized positions, queried as ground truth from the env -- not
sensed). Reports success rate, time-to-goal (successes only), path efficiency (straight-line
distance / actual distance traveled -- 1.0 is perfectly direct), and -- with obstacles enabled --
closest approach to any obstacle's surface and a collision rate (fraction of episodes where that
approach went negative, i.e. inside the obstacle).

Usage:
    python eval_navigate.py --episodes 32
    python eval_navigate.py --episodes 32 --terrain-amplitude 0.0 0.08
    python eval_navigate.py --episodes 32 --nav-obstacles 4
"""
import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv
from eval_policy import latest_checkpoint
from navigate import compute_nav_command


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to the segment a-b (not the infinite line)."""
    abx, aby = bx - ax, by - ay
    length2 = abx ** 2 + aby ** 2
    if length2 < 1e-9:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length2))
    cx, cy = ax + t * abx, ay + t * aby
    return float(np.hypot(px - cx, py - cy))


def make_env(seed, env_kwargs):
    def _init():
        env = Go1FlatEnv(render_mode=None, domain_randomize=True, **env_kwargs)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model_path, vecnorm_path, env_kwargs, episodes, seconds, kp, tolerance,
             goal_dist_range, speed_lo, speed_hi, lat_lo, lat_hi,
             obstacle_gain=1.5, obstacle_influence=0.8, seed0=20_000):
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
    # Each sub-env independently randomized its own obstacle_positions inside this reset() call
    # (if nav_obstacles > 0) -- query them once now (static for the whole episode), one get_attr
    # per attribute name, each returning a list with one value PER sub-env (unlike set_attr, which
    # broadcasts the SAME value to every targeted index -- get_attr collects, it doesn't broadcast).
    obstacle_radius = env.get_attr("nav_obstacle_radius")
    obstacles_per_ep = [[(ox, oy, obstacle_radius[i]) for ox, oy in pos]
                         for i, pos in enumerate(env.get_attr("obstacle_positions"))]
    closest_clearance = np.full(episodes, np.inf)
    # Whether a NAIVE straight-line path (the start -> goal segment, ignoring avoidance entirely)
    # would have come within (radius + robot's own rough half-width) of any obstacle -- tags which
    # episodes actually exercise the avoidance logic at all, since obstacles scattered uniformly
    # over a wide area often land nowhere near a particular episode's own short goal segment (most
    # of this project's eval numbers would otherwise silently average in a lot of "irrelevant
    # obstacle" episodes that say nothing about avoidance competence specifically).
    robot_half_width = 0.3
    path_blocked = np.array([
        any(_point_to_segment_dist(ox, oy, 0.0, 0.0, goal_x[i], goal_y[i]) < r + robot_half_width
            for ox, oy, r in obstacles_per_ep[i])
        for i in range(episodes)
    ])

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
    for i in range(episodes):
        for ox, oy, r in obstacles_per_ep[i]:
            closest_clearance[i] = min(closest_clearance[i], float(np.hypot(x[i] - ox, y[i] - oy)) - r)

    max_steps = int(seconds * 50)
    done_ep = np.zeros(episodes, dtype=bool)
    res = [None] * episodes
    for step in range(2, max_steps + 1):
        dist = np.hypot(goal_x - x, goal_y - y)
        reached_now = dist <= tolerance
        for i in range(episodes):
            if done_ep[i]:
                continue
            if reached_now[i]:
                fwd_cmd, lat_cmd = 0.0, 0.0
            else:
                fwd_cmd, lat_cmd = compute_nav_command(
                    x[i], y[i], yaw[i], goal_x[i], goal_y[i], obstacles_per_ep[i], kp,
                    speed_lo, speed_hi, lat_lo, lat_hi, obstacle_gain, obstacle_influence)
            env.env_method("set_command", fwd_cmd, lat_cmd, indices=[i])

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        new_x = np.array([inf["base_pos"][0] for inf in infos])
        new_y = np.array([inf["base_pos"][1] for inf in infos])
        path_len += np.hypot(new_x - x, new_y - y)
        x, y = new_x, new_y
        yaw = np.array([inf["yaw"] for inf in infos])
        for i in range(episodes):
            for ox, oy, r in obstacles_per_ep[i]:
                closest_clearance[i] = min(closest_clearance[i], float(np.hypot(x[i] - ox, y[i] - oy)) - r)

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
    has_obstacles = any(obstacles_per_ep)
    result = dict(
        episodes=episodes, success_rate=float(reached.mean()), fall_rate=float(fell.mean()),
        time_to_goal_mean=float(t_reach.mean()) if t_reach.size else float("nan"),
        path_efficiency_mean=float(eff.mean()) if eff.size else float("nan"),
    )
    if has_obstacles:
        result["collision_rate"] = float((closest_clearance < 0).mean())
        result["closest_clearance_mean"] = float(closest_clearance.mean())
        result["n_blocked"] = int(path_blocked.sum())
        if path_blocked.any():
            result["blocked_success_rate"] = float(reached[path_blocked].mean())
            result["blocked_collision_rate"] = float((closest_clearance[path_blocked] < 0).mean())
    return result


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
    ap.add_argument("--nav-obstacles", type=int, default=0,
                    help="Scatter this many static, impassable cylinder obstacles in the course "
                         "(Stage 10b) and steer around them. 0 (default) = none.")
    ap.add_argument("--nav-obstacle-radius", type=float, default=0.15)
    ap.add_argument("--nav-obstacle-height", type=float, default=0.5)
    ap.add_argument("--obstacle-avoid-gain", type=float, default=1.5)
    ap.add_argument("--obstacle-influence-radius", type=float, default=0.8)
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
                  obstacle_height_range=None, obstacle_height=None,
                  nav_obstacles=args.nav_obstacles, nav_obstacle_radius=args.nav_obstacle_radius,
                  nav_obstacle_height=args.nav_obstacle_height)
        e = evaluate(model_path, vec, kw, args.episodes, args.seconds, args.kp, args.tolerance,
                     args.goal_distance_range, speed_lo, speed_hi, lat_lo, lat_hi,
                     args.obstacle_avoid_gain, args.obstacle_influence_radius)
        terr = "flat" if amp is None else f"terrain {amp * 100:.0f} cm"
        line = (f"{terr:>13s}: success_rate={e['success_rate']:.2f}  fall_rate={e['fall_rate']:.2f}  "
                f"time_to_goal={e['time_to_goal_mean']:5.1f}s  path_efficiency={e['path_efficiency_mean']:.2f}")
        if "collision_rate" in e:
            line += (f"  collision_rate={e['collision_rate']:.2f}  "
                     f"closest_clearance={e['closest_clearance_mean']:.2f}m")
        print(line, flush=True)
        if "collision_rate" in e:
            n = e["n_blocked"]
            if n:
                print(f"{'':>13s}  of which {n}/{e['episodes']} episodes had an obstacle actually on "
                      f"the direct path: success_rate={e['blocked_success_rate']:.2f}  "
                      f"collision_rate={e['blocked_collision_rate']:.2f}", flush=True)
            else:
                print(f"{'':>13s}  (no episode had an obstacle actually on the direct path -- "
                      f"try --nav-obstacles higher or a smaller --goal-distance-range)", flush=True)


if __name__ == "__main__":
    main()
