"""Demo GIFs for visual inspection of each gait style (trot/pace/bound/walk), zero
residual (pure MPC, matching the Stage 11 MPC-only pivot -- no RL involved in gait-style
selection). Same tracking-camera convention as demo_trained_residual.py. Saved to
media/mpc_gait_<style>.gif and embedded in mpc/README.md.

Forward walking at a fixed moderate speed, flat ground, no residual action -- isolates
the gait-style effect itself, not a trained policy's compensation for it. Bound is
expected (and known, see README/HISTORY) to fail; the episode is left to run to
termination so the failure mode itself is visible, not cut short.
"""
import os
os.environ["MPLBACKEND"] = "Agg"

import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpc_residual_env import MPCResidualEnv

SECONDS = 5.0
MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"

for style in ("trot", "pace", "bound", "walk"):
    env = MPCResidualEnv(
        gait_style=style, episode_seconds=SECONDS,
        speed_range=(0.5, 0.5), lateral_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0),
        friction_range=(0.9, 0.9), mass_scale_range=(1.0, 1.0),
    )
    obs, _ = env.reset(seed=0)

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj.mj_name2id(env.mujoco_go2.model, mj.mjtObj.mjOBJ_BODY, "base_link")
    cam.distance = 3.0
    cam.elevation = -25
    cam.azimuth = 90

    renderer = mj.Renderer(env.mujoco_go2.model, height=270, width=360)
    frames = []
    for i in range(env.ctrl_steps):
        obs, reward, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
        if i % 4 == 0:
            renderer.update_scene(env.mujoco_go2.data, camera=cam)
            frames.append(Image.fromarray(renderer.render().copy()))
        if term or trunc:
            print(f"  style={style}: episode ended at step {i} (terminated={term})")
            break

    out_gif = MEDIA_DIR / f"mpc_gait_{style}.gif"
    # optimize=True is deliberately NOT used -- this Pillow version corrupts AND bloats
    # multi-frame GIF output with it (confirmed directly: identical frames gave a
    # corrupted 11.1MB file with optimize=True vs. a correct 2.6MB file without).
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=67, loop=0)
    print(f"style={style}: saved {len(frames)} frames to {out_gif} "
          f"({out_gif.stat().st_size / 1e6:.1f} MB)")
    # Without this, mj.Renderer's offscreen GL context from one loop iteration bleeds
    # into the next (confirmed directly: the FIRST style in the loop always rendered
    # correctly, every subsequent one came out corrupted) -- close explicitly before
    # the next iteration creates a new Renderer.
    renderer.close()
