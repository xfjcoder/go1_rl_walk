"""
Stage 11 step 3: verify the RL-residual interface (mpc_residual_env.MPCResidualEnv) is wired
correctly. The actual check: with the residual action held at ZERO every step, the achieved
yaw-rate tracking must reproduce the already-verified pure-MPC result from verify_turning.py
(1.91 rad/s in-place, 1.93 rad/s combined walk+turn, against a 2.0 rad/s command) -- if the
residual plumbing is correct, adding a zero residual changes nothing.
"""
import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from mpc_residual_env import MPCResidualEnv

env = MPCResidualEnv(
    yaw_rate_range=(2.0, 2.0),   # force the command, matching ex00_demo's rotation phase
    speed_range=(0.0, 0.0),
    lateral_range=(0.0, 0.0),
    friction_range=(0.9, 0.9),   # no randomization for this check -- isolate the interface itself
    mass_scale_range=(1.0, 1.0),
    episode_seconds=4.0,
    seed=0,
)

obs, info = env.reset(seed=0)
zero_action = np.zeros(12, dtype=np.float32)

wz_log = []
fell = False
for step in range(env.ctrl_steps):
    obs, reward, terminated, truncated, info = env.step(zero_action)
    wz_log.append(info["w_body"][2])
    if terminated:
        fell = True
        print(f"FELL at step {step} (base_height={info['base_height']:.3f}, "
              f"roll={info['roll']:.2f}, pitch={info['pitch']:.2f})")
        break
    if truncated:
        break

wz_log = np.array(wz_log)
print(f"steps run: {len(wz_log)}/{env.ctrl_steps}  fell={fell}")
if len(wz_log) > 50:
    # skip the first ~0.5s (50 ticks) spin-up transient, matching verify_turning.py's own
    # practice of only averaging over steady-state portions of a phase
    steady = wz_log[50:]
    print(f"mean achieved wz (steady-state): {steady.mean():+.3f} rad/s  "
          f"(cmd=2.0 rad/s; pure-MPC reference was +1.91 rad/s)")
    print(f"std: {steady.std():.3f} rad/s")
else:
    print("too few steps to report steady-state mean")
