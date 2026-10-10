"""
Stage 11 (MPC exploration), step 2: run third_party/go2-convex-mpc's own command-scheduled
walk+turn demo (its examples/ex00_demo.py), but driving THIS PROJECT's own Go2 model
(mpc/go2_mesh_for_mpc.xml -- a cosmetically-adapted copy of ../assets/go2_mesh.xml, see
that file's own header comment for exactly what and why) instead of the submodule's bundled
one, and without the interactive viewer (mjv.launch_passive needs a real display; this
project's own sandboxed dev environment has none -- same constraint navigate.py's own
render_mode="human" path runs into, same workaround: offscreen mujoco.Renderer instead).

Requires a SEPARATE venv from this project's own (see mpc/README.md) -- the submodule pins
numpy<2 and python<3.11, this project's venv is numpy>=2/python3.12. Run from mpc/:
    source <mpc-venv>/bin/activate
    python3 verify_turning.py

Prints a summary of commanded vs. achieved yaw rate during the schedule's two turning
phases (4-6s pure rotation, 6.5-8s walk+turn), and saves a trajectory plot + a GIF
(offscreen-rendered, tracking camera) to media/.
"""
import os
os.environ["MPLBACKEND"] = "Agg"   # headless sandbox, no display

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco as mj
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE_SRC = REPO_ROOT / "third_party" / "go2-convex-mpc" / "src"
sys.path.insert(0, str(SUBMODULE_SRC))

from convex_mpc.go2_robot_data import PinGo2Model
import convex_mpc.mujoco_model as mujoco_model_module

# Point the simulated plant at THIS PROJECT's own (adapted) Go2 model instead of the
# submodule's bundled one. The submodule's own Pinocchio reference model (PinGo2Model,
# used for the actual MPC dynamics) stays untouched -- only the MuJoCo-simulated robot
# being driven changes.
mujoco_model_module.XML_PATH = Path(__file__).resolve().parent / "go2_mesh_for_mpc.xml"

from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.leg_controller import LegController
from convex_mpc.gait import Gait

# --------------------------------------------------------------------------------
# Parameters (mirrors third_party/go2-convex-mpc/examples/ex00_demo.py exactly)
# --------------------------------------------------------------------------------
INITIAL_X_POS = -5
INITIAL_Y_POS = 0
RUN_SIM_LENGTH_S = 10.0

RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ


@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float


# [start_time (s), end_time (s), x_velocity (m/s), y_velocity (m/s), z_position (m),
#  yaw_angular_velocity (rad/s)]
CMD_SCHEDULE = [
    BodyCmdPhase(0.0, 1.0, 0.7, 0.0, 0.27, 0.0),   # Forward 0.7 m/s
    BodyCmdPhase(1.0, 1.5, 0.0, 0.0, 0.27, 0.0),   # Stop
    BodyCmdPhase(1.5, 3.0, 0.0, 0.3, 0.27, 0.0),   # Sideway 0.3 m/s
    BodyCmdPhase(3.0, 4.0, 0.0, 0.0, 0.27, 0.0),   # Stop
    BodyCmdPhase(4.0, 6.0, 0.0, 0.0, 0.27, 2.0),   # Rotate 2 rad/s
    BodyCmdPhase(6.0, 6.5, 0.0, 0.0, 0.27, 0.0),   # Stop
    BodyCmdPhase(6.5, 8.0, 0.6, 0.0, 0.27, 2.0),   # Forward 0.6 m/s + Rotate 2 rad/s
    BodyCmdPhase(8.0, 9.0, 0.8, 0.0, 0.27, 0.0),   # Forward 0.8 m/s
    BodyCmdPhase(9.0, 10.0, 0.0, 0.0, 0.27, 0.0),  # Stop
]

GAIT_HZ = 3
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

x_vel_des_body = 0.0
y_vel_des_body = 0.0
z_pos_des_body = 0.27
yaw_rate_des_body = 0.0

SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ
SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

HIP_LIM, ABD_LIM, KNEE_LIM, SAFETY = 23.7, 23.7, 45.43, 0.9
TAU_LIM = SAFETY * np.array([HIP_LIM, ABD_LIM, KNEE_LIM] * 4)

LEG_SLICE = {"FL": slice(0, 3), "FR": slice(3, 6), "RL": slice(6, 9), "RR": slice(9, 12)}


def get_body_cmd(t: float):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase.x_vel, phase.y_vel, phase.z_pos, phase.yaw_rate
    return 0.0, 0.0, 0.27, 0.0


# Centroidal state x = [px, py, pz, r, p, y, vx, vy, vz, wx, wy, wz]
x_vec = np.zeros((12, CTRL_STEPS))
mpc_force_world = np.zeros((12, CTRL_STEPS))
tau_raw = np.zeros((12, CTRL_STEPS))
tau_cmd = np.zeros((12, CTRL_STEPS))

go2 = PinGo2Model()
mujoco_go2 = MuJoCo_GO2_Model()
leg_controller = LegController()
traj = ComTraj(go2)
gait = Gait(GAIT_HZ, GAIT_DUTY)

q_init = go2.current_config.get_q()
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
mujoco_go2.update_with_q_pin(q_init)
mujoco_go2.model.opt.timestep = SIM_DT

traj.generate_traj(go2, gait, 0.0, x_vel_des_body, y_vel_des_body, z_pos_des_body,
                    yaw_rate_des_body, time_step=MPC_DT)
mpc = CentroidalMPC(go2, traj)
U_opt = np.zeros((12, traj.N), dtype=float)

time_log_render, q_log_render = [], []
next_render_t = 0.0

print(f"Running simulation for {RUN_SIM_LENGTH_S}s (model: {mujoco_model_module.XML_PATH})")
sim_start_time = time.perf_counter()
ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)

for k in range(SIM_STEPS):
    time_now_s = float(mujoco_go2.data.time)

    if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
        x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body = get_body_cmd(time_now_s)
        mujoco_go2.update_pin_with_mujoco(go2)
        x_vec[:, ctrl_i] = go2.compute_com_x_vec().reshape(-1)

        if (ctrl_i % STEPS_PER_MPC) == 0:
            print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)
            traj.generate_traj(go2, gait, time_now_s, x_vel_des_body, y_vel_des_body,
                                z_pos_des_body, yaw_rate_des_body, time_step=MPC_DT)
            sol = mpc.solve_QP(go2, traj, False)
            N = traj.N
            w_opt = sol["x"].full().flatten()
            U_opt = w_opt[12 * N:].reshape((12, N), order="F")

        mpc_force_world[:, ctrl_i] = U_opt[:, 0]
        for leg in ("FL", "FR", "RL", "RR"):
            res = leg_controller.compute_leg_torque(leg, go2, gait, mpc_force_world[LEG_SLICE[leg], ctrl_i], time_now_s)
            tau_raw[LEG_SLICE[leg], ctrl_i] = res.tau
        tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)
        tau_hold = tau_cmd[:, ctrl_i].copy()
        ctrl_i += 1

    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
    mujoco_go2.set_joint_torque(tau_hold)
    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    t_after = float(mujoco_go2.data.time)
    if t_after + 1e-12 >= next_render_t:
        time_log_render.append(t_after)
        q_log_render.append(mujoco_go2.data.qpos.copy())
        next_render_t += RENDER_DT

print(f"\nSimulation ended. Control ticks: {ctrl_i}/{CTRL_STEPS}  "
      f"(elapsed {time.perf_counter() - sim_start_time:.1f}s)")

# --------------------------------------------------------------------------------
# Summary: commanded vs. achieved yaw rate during the two turning phases
# --------------------------------------------------------------------------------
t_vec = np.arange(ctrl_i) * CTRL_DT
yaw = x_vec[5, :ctrl_i]
wz = x_vec[11, :ctrl_i]
px, py = x_vec[0, :ctrl_i], x_vec[1, :ctrl_i]


def avg_over(t0, t1, arr):
    mask = (t_vec >= t0) & (t_vec < t1)
    return float(np.mean(arr[mask])) if mask.any() else float("nan")


print("\n--- summary ---")
print(f"final pos: x={px[-1]:+.3f} y={py[-1]:+.3f}  final yaw={np.degrees(yaw[-1]):+.1f} deg")
print(f"mean wz during 4-6s pure rotation   (cmd=2.0 rad/s): {avg_over(4.0, 6.0, wz):+.3f} rad/s")
print(f"mean wz during 6.5-8.0s walk+turn   (cmd=2.0 rad/s): {avg_over(6.5, 8.0, wz):+.3f} rad/s")
print(f"mean wz during 0-1.0s pure forward  (cmd=0.0 rad/s): {avg_over(0.0, 1.0, wz):+.3f} rad/s")

MEDIA_DIR = REPO_ROOT / "media"
MEDIA_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(8, 9))
axes[0].plot(px, py); axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
axes[0].set_title("base XY trajectory"); axes[0].axis("equal")
axes[1].plot(t_vec, np.degrees(yaw)); axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("yaw (deg)")
axes[1].set_title("base yaw over time")
axes[2].plot(t_vec, wz); axes[2].set_xlabel("t (s)"); axes[2].set_ylabel("wz (rad/s)")
axes[2].set_title("yaw rate over time")
fig.tight_layout()
fig.savefig(MEDIA_DIR / "mpc_turning_summary.png", dpi=100)
print(f"saved plot to {MEDIA_DIR / 'mpc_turning_summary.png'}")

# --------------------------------------------------------------------------------
# Offscreen GIF (no interactive viewer in this sandbox -- replay the logged render-rate
# qpos trace through a fresh MjData + mujoco.Renderer instead of
# mujoco_go2.replay_simulation()'s mjv.launch_passive, which needs a real display)
# --------------------------------------------------------------------------------
from PIL import Image

render_model = mujoco_go2.model
render_data = mj.MjData(render_model)
renderer = mj.Renderer(render_model, height=480, width=640)

cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mj.mj_name2id(render_model, mj.mjtObj.mjOBJ_BODY, "base_link")
cam.distance = 3.0
cam.elevation = -25
cam.azimuth = 90

frames = []
FRAME_STRIDE = 2  # 120Hz render log -> 60 fps gif
for i in range(0, len(q_log_render), FRAME_STRIDE):
    render_data.qpos[:] = q_log_render[i]
    mj.mj_forward(render_model, render_data)
    renderer.update_scene(render_data, camera=cam)
    frames.append(Image.fromarray(renderer.render().copy()))

out_gif = MEDIA_DIR / "mpc_turning_demo.gif"
frames[0].save(out_gif, save_all=True, append_images=frames[1:],
               duration=1000.0 / (RENDER_HZ / FRAME_STRIDE), loop=0)
print(f"saved {len(frames)} frames to {out_gif}")
