"""
Load a trained checkpoint and watch Go1 walk.

Usage:
    python play.py --run-dir runs/pilot_a_kp40 --record out.gif   # latest checkpoint of a run, with the run's own
                                                                   # env settings (kp/kd/rewards) -- preferred
    python play.py --run-dir runs/pilot_a_kp40 --model go1_flat_1000000_steps --record out.gif   # a specific one
    python play.py --model checkpoints/go1_flat_final.zip                # interactive viewer (needs GLFW)
    python play.py --model checkpoints/go1_flat_final.zip --record out.gif  # offscreen -> GIF, no window

Note: the interactive viewer uses GLFW, whose Wayland backend segfaults
on some Linux setups. If the plain command crashes, try:
    WAYLAND_DISPLAY= python3 play.py --model ...
or use --record, which renders offscreen and never opens a window.
"""
import argparse
import json
import os
import re
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv
from eval_policy import latest_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                         help="runs/<name> from train.py --run-name. Uses that run's env_kwargs.json and "
                              "its latest checkpoint (or --model NAME inside <run-dir>/checkpoints).")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnormalize", type=str, default="checkpoints/vecnormalize_final.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--terrain-amplitude", type=float, default=None,
                         help="Play on rough terrain of this amplitude (m, peak-to-peak).")
    parser.add_argument("--slope-deg", type=float, default=None,
                         help="Play on a slope of this angle (deg, signed: + uphill, - downhill).")
    parser.add_argument("--target-speed", type=float, default=None,
                         help="Command speed (m/s). Default: the run's own with --run-dir, else 1.0.")
    parser.add_argument("--record", type=str, default=None,
                         help="Save an offscreen-rendered GIF instead of opening a live viewer")
    parser.add_argument("--slowmo", type=float, default=1.0,
                         help="Playback speed multiplier for --record, e.g. 0.5 = half speed, "
                              "0.25 = quarter speed. Stretches frame duration; doesn't drop frames.")
    parser.add_argument("--frame-stride", type=int, default=2,
                         help="Keep every Nth rendered frame for --record (lower = smoother but bigger file).")
    parser.add_argument("--camera", type=str, default="track", choices=["track", "chase_rear", "topdown"],
                         help="track: side view. chase_rear: follows from behind, best for spotting "
                              "left/right leg asymmetry. topdown: best for footfall timing patterns.")
    args = parser.parse_args()

    render_mode = "rgb_array" if args.record else "human"

    env_kwargs = {}
    if args.run_dir:
        ckpt_dir = os.path.join(args.run_dir, "checkpoints")
        model_path = os.path.join(ckpt_dir, args.model) if args.model else latest_checkpoint(ckpt_dir)
        m = re.search(r"go1_flat_(\d+)_steps$", os.path.basename(model_path))
        args.vecnormalize = os.path.join(ckpt_dir, f"go1_flat_vecnormalize_{m.group(1)}_steps.pkl") if m \
            else os.path.join(ckpt_dir, "vecnormalize_final.pkl")
        with open(os.path.join(args.run_dir, "env_kwargs.json")) as f:
            env_kwargs = json.load(f)
        print(f"Using {model_path}.zip with the run's env settings (kp={env_kwargs['kp']}, kd={env_kwargs['kd']})")
    else:
        if not args.model:
            parser.error("give --run-dir or --model")
        model_path = args.model
    if args.target_speed is not None:
        env_kwargs["target_speed"] = args.target_speed
    env_kwargs["command_speed_range"] = None      # play at one fixed command (--target-speed, else the run's)
    env_kwargs["terrain_amplitude_range"] = None
    env_kwargs["terrain_amplitude"] = args.terrain_amplitude
    env_kwargs["slope_range"] = None
    env_kwargs["slope_deg"] = args.slope_deg
    env_kwargs.setdefault("target_speed", 1.0)

    def make_env():
        return Go1FlatEnv(render_mode=render_mode, domain_randomize=False, camera=args.camera, **env_kwargs)

    env = DummyVecEnv([make_env])
    try:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("No VecNormalize stats found, running without observation normalization.")

    model = PPO.load(model_path)

    frames = []
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward[0]
                if args.record:
                    frame = env.envs[0].render(camera=args.camera)
                    if frame is not None:
                        frames.append(frame)
                else:
                    time.sleep(0.02)  # ~50Hz real-time playback
            print(f"Episode {ep + 1}: reward = {ep_reward:.1f}")
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping and saving anything already recorded...")
    finally:
        if args.record and frames:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames[::args.frame_stride]]
            base_fps = 50.0 / args.frame_stride  # sim runs at 50Hz; stride thins it down
            frame_duration_ms = (1000.0 / base_fps) / args.slowmo
            imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=frame_duration_ms, loop=0)
            print(f"Saved {len(imgs)} frames to {args.record} "
                  f"(~{frame_duration_ms:.1f}ms/frame, {args.slowmo}x speed)")
        env.close()


if __name__ == "__main__":
    main()

