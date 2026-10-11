"""
Stage 11 step 4: demo GIF of the TRAINED residual policy driving the MPC-controlled
robot, for visual comparison against the pure-MPC (zero-residual) turning demo in
mpc/README.md (media/mpc_turning_demo.gif).

Same command schedule style as verify_turning.py's own in-place-rotation test (forced
yaw-rate command), but stepping through MPCResidualEnv directly (not the raw submodule
demo script) so the trained policy's own residual actions are actually applied.
"""
import os
os.environ["MPLBACKEND"] = "Agg"

import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpc_residual_env import MPCResidualEnv

RUN_DIR = Path(__file__).resolve().parent / "runs" / "v1"
SECONDS = 10.0

env = DummyVecEnv([lambda: MPCResidualEnv(
    episode_seconds=SECONDS,
    yaw_rate_range=(2.0, 2.0),
    speed_range=(0.0, 0.0),
    lateral_range=(0.0, 0.0),
    friction_range=(0.9, 0.9),
    mass_scale_range=(1.0, 1.0),
)])
env = VecNormalize.load(str(RUN_DIR / "checkpoints" / "vecnormalize_final.pkl"), env)
env.training = False
env.norm_reward = False
model = PPO.load(str(RUN_DIR / "checkpoints" / "final.zip"))
raw_env = env.envs[0]

env.seed(0)
obs = env.reset()
cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mj.mj_name2id(raw_env.mujoco_go2.model, mj.mjtObj.mjOBJ_BODY, "base_link")
cam.distance = 3.0
cam.elevation = -25
cam.azimuth = 90

renderer = mj.Renderer(raw_env.mujoco_go2.model, height=480, width=640)
frames = []
wz_log = []
ctrl_steps = raw_env.ctrl_steps
for i in range(ctrl_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    wz_log.append(info[0]["w_body"][2])
    if i % 3 == 0:
        renderer.update_scene(raw_env.mujoco_go2.data, camera=cam)
        frames.append(Image.fromarray(renderer.render().copy()))
    if done[0]:
        print(f"episode ended at step {i}: {info[0]}")
        break

wz_log = np.array(wz_log)
print(f"ran {len(wz_log)}/{ctrl_steps} steps")
if len(wz_log) > 50:
    print(f"mean achieved wz (steady-state, skip first 0.25s): {wz_log[50:].mean():+.3f} rad/s (cmd=2.0)")

out_gif = Path(__file__).resolve().parent.parent / "media" / "mpc_residual_trained_demo.gif"
frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=50, loop=0)
print(f"saved {len(frames)} frames to {out_gif}")
