"""
Load a trained checkpoint and watch Go1 walk.

Usage:
    python play.py --model checkpoints/go1_flat_final.zip                # interactive viewer (needs GLFW)
    python play.py --model checkpoints/go1_flat_final.zip --record out.gif  # offscreen -> GIF, no window

Note: the interactive viewer uses GLFW, whose Wayland backend segfaults
on some Linux setups. If the plain command crashes, try:
    WAYLAND_DISPLAY= python3 play.py --model ...
or use --record, which renders offscreen and never opens a window.
"""
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.go1_env import Go1FlatEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, default="checkpoints/vecnormalize_final.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--target-speed", type=float, default=1.0)
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

    def make_env():
        return Go1FlatEnv(render_mode=render_mode, target_speed=args.target_speed,
                           domain_randomize=False, camera=args.camera)

    env = DummyVecEnv([make_env])
    try:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("No VecNormalize stats found, running without observation normalization.")

    model = PPO.load(args.model)

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

