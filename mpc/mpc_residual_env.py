"""
Stage 11 step 3: the RL-residual interface over third_party/go2-convex-mpc.

MPC (CentroidalMPC + LegController, unmodified, imported straight from the submodule)
computes a nominal 12-dim joint torque command each control tick exactly as it already
does in verify_turning.py. This env's ONLY addition is: an RL policy observes the
robot's state (+ the MPC's own nominal torque, for context) and outputs a small per-
joint torque RESIDUAL, added to the MPC's command before it's clipped to the actuator
limits and applied. MPC still does 100% of the locomotion planning (footstep timing,
swing trajectories, stance force allocation, turning) -- nothing about its own logic is
touched. The residual's job (scoped the same way every other RL stage in this project
has been scoped): compensate for domain randomization / disturbances the MPC's
simplified rigid-body model doesn't explicitly plan for, not replace what MPC already
does well.

A zero residual (action=0 every step) must reproduce the already-verified pure-MPC
behavior (see mpc/README.md: 1.91/1.93 rad/s achieved turning) -- that's the actual
correctness check for this interface, exercised by this file's own __main__ block.
"""
import sys
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE_SRC = REPO_ROOT / "third_party" / "go2-convex-mpc" / "src"
sys.path.insert(0, str(SUBMODULE_SRC))

from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import Gait
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.leg_controller import LegController
import convex_mpc.mujoco_model as mujoco_model_module
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
import pinocchio as pin

DEFAULT_XML = Path(__file__).resolve().parent / "go2_mesh_for_mpc.xml"

GAIT_HZ = 3
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

SIM_HZ = 1000
CTRL_HZ = 200
CTRL_DECIM = SIM_HZ // CTRL_HZ
MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

HIP_LIM, ABD_LIM, KNEE_LIM, SAFETY = 23.7, 23.7, 45.43, 0.9
TAU_LIM = SAFETY * np.array([HIP_LIM, ABD_LIM, KNEE_LIM] * 4)
LEG_SLICE = {"FL": slice(0, 3), "FR": slice(3, 6), "RL": slice(6, 9), "RR": slice(9, 12)}

STAND_HEIGHT = 0.27
FALL_HEIGHT = 0.15       # terminate if base drops below this (m)
FALL_TILT = 0.9          # terminate if |roll| or |pitch| exceeds this (rad, ~51deg)


class MPCResidualEnv(gym.Env):
    def __init__(
        self,
        xml_path: str = str(DEFAULT_XML),
        residual_scale: float = 5.0,         # N*m: action in [-1,1] -> +-residual_scale
        residual_effort_weight: float = 0.002,
        speed_range: tuple = (-0.4, 0.8),
        lateral_range: tuple = (-0.3, 0.3),
        yaw_rate_range: tuple = (-1.0, 1.0),
        friction_range: tuple = (0.6, 1.1),
        mass_scale_range: tuple = (0.8, 1.2),
        push_velocity: float = 0.0,           # m/s: random horizontal velocity kick, 0 = off
        episode_seconds: float = 10.0,
        speed_tracking_sigma: float = 0.5,
        yaw_rate_tracking_sigma: float = 0.5,
        seed: int | None = None,
    ):
        self.xml_path = xml_path
        self.residual_scale = residual_scale
        self.residual_effort_weight = residual_effort_weight
        self.speed_range = speed_range
        self.lateral_range = lateral_range
        self.yaw_rate_range = yaw_rate_range
        self.friction_range = friction_range
        self.mass_scale_range = mass_scale_range
        self.push_velocity = push_velocity
        self.episode_seconds = episode_seconds
        self.speed_tracking_sigma = speed_tracking_sigma
        self.yaw_rate_tracking_sigma = yaw_rate_tracking_sigma
        self.ctrl_steps = int(episode_seconds * CTRL_HZ)

        self._rng = np.random.default_rng(seed)

        # obs = [v_body(3), w_body(3), roll, pitch, qpos(12), qvel(12), mpc_tau(12), cmd(3)] = 47
        obs_dim = 3 + 3 + 2 + 12 + 12 + 12 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        # Built ONCE here, not per-reset(): PinGo2Model() and ComTraj() each cost ~1.2s
        # to construct (confirmed by direct timing -- re-parsing the URDF / building
        # CasADi's symbolic expression graph, not a one-time process-wide cost, genuinely
        # ~1.2s on EVERY instantiation), which would make reset() cost ~2.4s and real
        # training impractical. reset() below only resets STATE on these same objects
        # (regenerates the trajectory for the new episode's command, resets MuJoCo data) --
        # confirmed safe: ComTraj.generate_traj()'s only coupling to go2 is reading its
        # CURRENT state each call (not accumulating hidden state of its own), and
        # update_pin_with_mujoco() re-syncs go2's state from mujoco_go2 every step anyway.
        mujoco_model_module.XML_PATH = Path(xml_path)   # see __init__ docstring note below --
        # MuJoCo_GO2_Model() reads this MODULE-LEVEL global, not a constructor argument;
        # forgetting this monkeypatch (an earlier bug, caught in review) silently loads
        # the submodule's own bundled model instead of this project's go2_mesh_for_mpc.xml.
        self.go2 = PinGo2Model()
        self.mujoco_go2 = MuJoCo_GO2_Model()
        self.leg_controller = LegController()
        self.gait = Gait(GAIT_HZ, GAIT_DUTY)
        self.traj = ComTraj(self.go2)
        self.mujoco_go2.model.opt.timestep = 1.0 / SIM_HZ
        self._q_init = self.go2.current_config.get_q().copy()
        self._floor_geom_id = mj.mj_name2id(self.mujoco_go2.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        self._base_id = mj.mj_name2id(self.mujoco_go2.model, mj.mjtObj.mjOBJ_BODY, "base_link")
        self._nominal_base_mass = float(self.mujoco_go2.model.body_mass[self._base_id])

        self.mpc = None
        self._ctrl_i = 0
        self._next_push_step = 10**9
        self.target_speed = 0.0
        self.target_lateral = 0.0
        self.target_yaw_rate = 0.0
        self._last_tau_cmd = np.zeros(12)

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.target_speed = float(self._rng.uniform(*self.speed_range))
        self.target_lateral = float(self._rng.uniform(*self.lateral_range))
        self.target_yaw_rate = float(self._rng.uniform(*self.yaw_rate_range))

        # Reset MuJoCo state fully (qpos/qvel/ctrl/warmstart etc. all zeroed, not just
        # qpos -- update_with_q_pin() alone only ever writes qpos, so without this any
        # leftover qvel from the PREVIOUS episode's final instant would silently carry
        # into the new one) before setting the standing qpos.
        mj.mj_resetData(self.mujoco_go2.model, self.mujoco_go2.data)
        self.mujoco_go2.update_with_q_pin(self._q_init.copy())

        # Domain randomization, mirroring envs/go1_env.py's own conventions: floor
        # friction and base mass scale sampled fresh each reset.
        mu = float(self._rng.uniform(*self.friction_range))
        if self._floor_geom_id >= 0:
            self.mujoco_go2.model.geom_friction[self._floor_geom_id, 0] = mu
        scale = float(self._rng.uniform(*self.mass_scale_range))
        self.mujoco_go2.model.body_mass[self._base_id] = self._nominal_base_mass * scale

        # Re-sync Pinocchio's state from the just-reset MuJoCo state (go2 is a
        # long-lived, reused object -- its internal state otherwise still reflects
        # wherever the PREVIOUS episode ended).
        self.mujoco_go2.update_pin_with_mujoco(self.go2)
        self.traj.generate_traj(self.go2, self.gait, 0.0, self.target_speed, self.target_lateral,
                                 STAND_HEIGHT, self.target_yaw_rate, time_step=MPC_DT)
        # CentroidalMPC() itself is cheap (~6ms, confirmed by direct timing) -- rebuilt
        # fresh each reset anyway, to avoid any question of stale OSQP warm-start state
        # leaking across episodes.
        self.mpc = CentroidalMPC(self.go2, self.traj)
        self._U_opt = np.zeros((12, self.traj.N), dtype=float)

        self._ctrl_i = 0
        self._next_push_step = (int(self._rng.integers(150, 300)) if self.push_velocity > 0 else 10**9)
        self._last_tau_cmd = np.zeros(12)

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        data = self.mujoco_go2.data
        qw, qx, qy, qz = data.qpos[3:7]
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        v_world = data.qvel[0:3]
        v_body = R.T @ v_world
        w_body = data.qvel[3:6]
        # roll/pitch from the rotation matrix (ZYX convention, consistent with go2.compute_com_x_vec)
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])
        qpos12 = data.qpos[7:19] - 0.0  # raw joint angles (no nominal offset subtracted -- kept simple)
        qvel12 = data.qvel[6:18]
        cmd = np.array([self.target_speed, self.target_lateral, self.target_yaw_rate])
        obs = np.concatenate([v_body, w_body, [roll, pitch], qpos12, qvel12, self._last_tau_cmd, cmd])
        return obs.astype(np.float32)

    def step(self, action):
        residual = np.clip(action, -1.0, 1.0) * self.residual_scale

        time_now_s = float(self.mujoco_go2.data.time)
        self.mujoco_go2.update_pin_with_mujoco(self.go2)

        if (self._ctrl_i % STEPS_PER_MPC) == 0:
            self.traj.generate_traj(self.go2, self.gait, time_now_s, self.target_speed,
                                     self.target_lateral, STAND_HEIGHT, self.target_yaw_rate,
                                     time_step=MPC_DT)
            sol = self.mpc.solve_QP(self.go2, self.traj, False)
            N = self.traj.N
            w_opt = sol["x"].full().flatten()
            self._U_opt = w_opt[12 * N:].reshape((12, N), order="F")

        mpc_force_world_0 = self._U_opt[:, 0]
        tau_raw = np.zeros(12)
        for leg in ("FL", "FR", "RL", "RR"):
            res = self.leg_controller.compute_leg_torque(
                leg, self.go2, self.gait, mpc_force_world_0[LEG_SLICE[leg]], time_now_s)
            tau_raw[LEG_SLICE[leg]] = res.tau
        tau_mpc = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        self._last_tau_cmd = tau_mpc.copy()

        tau_final = np.clip(tau_mpc + residual, -TAU_LIM, TAU_LIM)

        if self._ctrl_i == self._next_push_step:
            self.mujoco_go2.data.qvel[0:2] += self._rng.uniform(-self.push_velocity, self.push_velocity, 2)
            self._next_push_step += int(self._rng.integers(150, 300))

        for _ in range(CTRL_DECIM):
            mj.mj_step1(self.mujoco_go2.model, self.mujoco_go2.data)
            self.mujoco_go2.set_joint_torque(tau_final)
            mj.mj_step2(self.mujoco_go2.model, self.mujoco_go2.data)

        self._ctrl_i += 1

        obs = self._get_obs()
        v_body, w_body = obs[0:3], obs[3:6]
        roll, pitch = obs[6], obs[7]
        base_height = float(self.mujoco_go2.data.qpos[2])

        r_speed = np.exp(-((v_body[0] - self.target_speed) / self.speed_tracking_sigma) ** 2)
        r_lateral = np.exp(-((v_body[1] - self.target_lateral) / self.speed_tracking_sigma) ** 2)
        r_yaw_rate = np.exp(-((w_body[2] - self.target_yaw_rate) / self.yaw_rate_tracking_sigma) ** 2)
        r_effort = -self.residual_effort_weight * float(np.sum(residual ** 2))
        reward = 1.0 + r_speed + r_lateral + r_yaw_rate + r_effort

        fell = base_height < FALL_HEIGHT or abs(roll) > FALL_TILT or abs(pitch) > FALL_TILT
        terminated = bool(fell)
        truncated = bool(self._ctrl_i >= self.ctrl_steps)

        info = {
            "base_height": base_height, "roll": roll, "pitch": pitch,
            "v_body": v_body.copy(), "w_body": w_body.copy(),
            "r_speed": r_speed, "r_lateral": r_lateral, "r_yaw_rate": r_yaw_rate, "r_effort": r_effort,
        }
        return obs, float(reward), terminated, truncated, info
