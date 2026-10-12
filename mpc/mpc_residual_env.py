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
import re
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE_SRC = REPO_ROOT / "third_party" / "go2-convex-mpc" / "src"
sys.path.insert(0, str(SUBMODULE_SRC))
sys.path.insert(0, str(REPO_ROOT))

from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import Gait
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.leg_controller import LegController
import convex_mpc.mujoco_model as mujoco_model_module
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
import pinocchio as pin

# Reuse this project's own heightfield constants/conventions exactly (not re-derived) --
# see envs/go1_env.py for the full rationale of each.
from envs.go1_env import HF_CELL, HF_CX, HF_HALF_X, HF_HALF_Y, HF_ELEV, HF_PAD_START, HF_PAD_END

DEFAULT_XML = Path(__file__).resolve().parent / "go2_mesh_for_mpc.xml"
DEFAULT_RAMP_LENGTH = 8.0


def _inject_terrain_xml(xml: str) -> str:
    """xml with a random-heightfield geom injected -- same mechanism and constants as
    envs/go1_env.py's own _inject_terrain_xml (the floor plane drops 5cm as a safety net
    so it never coincides with the terrain surface; heights are always >= 0)."""
    nrow, ncol = int(round(2 * HF_HALF_Y / HF_CELL)) + 1, int(round(2 * HF_HALF_X / HF_CELL)) + 1
    edits = [
        ("</asset>", f'<hfield name="terrain" nrow="{nrow}" ncol="{ncol}" '
                     f'size="{HF_HALF_X} {HF_HALF_Y} {HF_ELEV} 0.1"/>\n  </asset>'),
        ('<geom name="floor" type="plane"', '<geom name="floor" pos="0 0 -0.05" type="plane"'),
        ('<light name="sun"', f'<geom name="terrain" type="hfield" hfield="terrain" pos="{HF_CX} 0 0" '
                              f'rgba="0.36 0.5 0.36 1" friction="0.9 0.02 0.01" condim="3"/>\n    <light name="sun"'),
    ]
    for old, new in edits:
        assert xml.count(old) == 1, f"cannot inject terrain: {old!r} not found exactly once"
        xml = xml.replace(old, new)
    return xml


def _fix_meshdir_absolute(xml: str, xml_dir: Path) -> str:
    """Rewrite a relative <compiler meshdir="..."/> to an absolute path -- needed once the
    terrain-injected XML gets written to a temp file elsewhere, decoupling it from
    go2_mesh_for_mpc.xml's own directory (where the relative meshdir would otherwise
    resolve from)."""
    return re.sub(
        r'(<compiler\b[^>]*\bmeshdir=")([^"]+)(")',
        lambda m: m.group(1) + str((xml_dir / m.group(2)).resolve()) + m.group(3),
        xml,
    )

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
        terrain_amplitude_range: tuple | None = None,  # (lo, hi) m peak-to-peak random bumps,
                                               # sampled each reset. None (default) = flat --
                                               # byte-identical to every pre-existing behavior.
        slope_deg_range: tuple | None = None,  # (lo, hi) degrees, sampled each reset; uphill/
                                               # downhill chosen with equal probability.
        stair_height_range: tuple | None = None,  # (lo, hi) m per riser, sampled each reset;
                                               # ascending/descending chosen with equal probability.
        use_terrain_heightmap: bool = False,  # add a 3x3 local heightmap (9 dims) to the
                                               # observation -- mirrors envs/go1_env.py's own
                                               # --use-terrain-heightmap exactly (a privileged
                                               # analytic lookup, zero on flat ground / terrain
                                               # disabled). Default False keeps obs_dim == 47.
        ramp_length: float = DEFAULT_RAMP_LENGTH,
        stair_depth: float = 0.25,
        num_stairs: int = 8,
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
        self.terrain_amplitude_range = terrain_amplitude_range
        self.slope_deg_range = slope_deg_range
        self.stair_height_range = stair_height_range
        self.terrain_enabled = any(r is not None for r in
                                    (terrain_amplitude_range, slope_deg_range, stair_height_range))
        # Curriculum ramp targets (mirrors envs/go1_env.py's own terrain_amp_max_current /
        # slope_deg_max_current / stair_height_max_current convention): each reset samples
        # U(range_lo, current_max) instead of the static range_hi. A RampCallback in
        # train_residual.py raises *_max_current from a curriculum-start value up to
        # range_hi over training; left untouched (== range_hi from construction) this is
        # byte-identical to uniform sampling across the full range, so omitting curriculum
        # flags reproduces terrain_v1's exact prior behavior.
        self.terrain_amp_max_current = terrain_amplitude_range[1] if terrain_amplitude_range else None
        self.slope_deg_max_current = slope_deg_range[1] if slope_deg_range else None
        self.stair_height_max_current = stair_height_range[1] if stair_height_range else None
        self.use_terrain_heightmap = use_terrain_heightmap
        self.ramp_length = ramp_length
        self.stair_depth = stair_depth
        self.num_stairs = num_stairs
        self.ctrl_steps = int(episode_seconds * CTRL_HZ)

        self._rng = np.random.default_rng(seed)

        # obs = [v_body(3), w_body(3), roll, pitch, qpos(12), qvel(12), mpc_tau(12), cmd(3)] = 47
        # [+ heightmap(9) if use_terrain_heightmap]
        obs_dim = 3 + 3 + 2 + 12 + 12 + 12 + 3 + (9 if use_terrain_heightmap else 0)
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
        if self.terrain_enabled:
            # Terrain-injected XML written to a fresh temp file (not go2_mesh_for_mpc.xml's own
            # directory) -- the meshdir gets rewritten to absolute first so it still resolves
            # from anywhere. Only built/used when at least one terrain range is actually given;
            # the flat (xml_path unchanged) case below is byte-identical to every pre-existing
            # behavior, including the already-trained v1 checkpoint's own flat-ground training.
            xml_text = Path(xml_path).read_text()
            xml_text = _fix_meshdir_absolute(xml_text, Path(xml_path).resolve().parent)
            xml_text = _inject_terrain_xml(xml_text)
            tmp_path = Path(tempfile.mkdtemp(prefix="mpc_terrain_")) / "go2_mesh_terrain.xml"
            tmp_path.write_text(xml_text)
            mujoco_model_module.XML_PATH = tmp_path
        else:
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

        self._terrain_geom = None
        self._terrain_h = None
        if self.terrain_enabled:
            hid = mj.mj_name2id(self.mujoco_go2.model, mj.mjtObj.mjOBJ_HFIELD, "terrain")
            self._hf_nrow = int(self.mujoco_go2.model.hfield_nrow[hid])
            self._hf_ncol = int(self.mujoco_go2.model.hfield_ncol[hid])
            self._hf_adr = int(self.mujoco_go2.model.hfield_adr[hid])
            self._hf_x0, self._hf_y0 = HF_CX - HF_HALF_X, -HF_HALF_Y
            self._terrain_geom = mj.mj_name2id(self.mujoco_go2.model, mj.mjtObj.mjOBJ_GEOM, "terrain")

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

        if self.terrain_enabled:
            amp = (float(self._rng.uniform(self.terrain_amplitude_range[0],
                                            max(self.terrain_amp_max_current, self.terrain_amplitude_range[0])))
                   if self.terrain_amplitude_range is not None else 0.0)
            if self.slope_deg_range is not None:
                slope_deg = float(self._rng.uniform(self.slope_deg_range[0],
                                                     max(self.slope_deg_max_current, self.slope_deg_range[0])))
                uphill = bool(self._rng.uniform() < 0.5)
            else:
                slope_deg, uphill = 0.0, True
            if self.stair_height_range is not None:
                stair_h = float(self._rng.uniform(self.stair_height_range[0],
                                                   max(self.stair_height_max_current, self.stair_height_range[0])))
                ascending = bool(self._rng.uniform() < 0.5)
            else:
                stair_h, ascending = 0.0, True
            self._generate_terrain(amp, slope_deg, uphill, stair_h, ascending)
            mu_terrain = mu  # same sampled friction as the floor, matching envs/go1_env.py's own
            self.mujoco_go2.model.geom_friction[self._terrain_geom, 0] = mu_terrain
            mj.mj_forward(self.mujoco_go2.model, self.mujoco_go2.data)

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

    def _generate_terrain(self, amp: float, slope_deg: float = 0.0, uphill: bool = True,
                           stair_h: float = 0.0, ascending: bool = True):
        """Ported from envs/go1_env.py's own _generate_terrain (same math, same HF_* constants) --
        discrete obstacles omitted (not in scope here; nav_obstacles already covers that case for
        the RL navigation stage). See that method's own docstring for the full derivation of each
        component; summary: random bumps + a constant-grade slope + stepped stairs, all additive,
        faded in only after a flat start pad so the robot always spawns on flat ground regardless
        of what's sampled for the rest of the course."""
        nrow, ncol = self._hf_nrow, self._hf_ncol
        xw = self._hf_x0 + np.arange(ncol) * HF_CELL

        if amp <= 1e-6:
            bumps = np.zeros((nrow, ncol))
        else:
            spacing = float(np.exp(self._rng.uniform(np.log(0.15), np.log(1.0))))
            xs, ys = np.arange(ncol) * HF_CELL / spacing, np.arange(nrow) * HF_CELL / spacing
            coarse = self._rng.uniform(0.0, 1.0, (int(ys[-1]) + 2, int(xs[-1]) + 2))
            i0 = np.floor(xs).astype(int)
            tmp = coarse[:, i0] * (1 - (xs - i0)) + coarse[:, i0 + 1] * (xs - i0)
            j0 = np.floor(ys).astype(int)
            wy = (ys - j0)[:, None]
            h01 = tmp[j0, :] * (1 - wy) + tmp[j0 + 1, :] * wy
            fade = np.clip((xw - HF_PAD_START) / (HF_PAD_END - HF_PAD_START), 0.0, 1.0)
            bumps = amp * h01 * (fade * fade * (3 - 2 * fade))[None, :]

        if slope_deg <= 1e-6:
            slope = np.zeros(ncol)
        else:
            rise = np.tan(np.radians(slope_deg)) * np.clip(xw - HF_PAD_END, 0.0, self.ramp_length)
            slope = rise if uphill else rise.max() - rise

        if stair_h <= 1e-6:
            stairs = np.zeros(ncol)
        else:
            step_num = np.clip(np.floor((xw - HF_PAD_END) / self.stair_depth), 0, self.num_stairs)
            rise = stair_h * step_num
            stairs = rise if ascending else rise.max() - rise

        h = np.clip(bumps + slope[None, :] + stairs[None, :], 0.0, HF_ELEV)
        self._terrain_h = h
        self.mujoco_go2.model.hfield_data[self._hf_adr:self._hf_adr + nrow * ncol] = (h / HF_ELEV).ravel()

    def _terrain_height(self, x, y):
        """Ported verbatim from envs/go1_env.py's own _terrain_height: bilinear-interpolated
        terrain surface height (m) at world (x, y); scalars or arrays. Zero on flat ground /
        terrain disabled (checked FIRST, before touching self._hf_x0 etc., which only exist
        when self.terrain_enabled -- safe to call even if use_terrain_heightmap=True without
        terrain_enabled, same as the reference implementation)."""
        if self._terrain_h is None:
            return np.zeros(np.shape(x)) if np.ndim(x) else 0.0
        fx = np.clip((np.asarray(x) - self._hf_x0) / HF_CELL, 0.0, self._hf_ncol - 1 - 1e-9)
        fy = np.clip((np.asarray(y) - self._hf_y0) / HF_CELL, 0.0, self._hf_nrow - 1 - 1e-9)
        c0, r0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        wx, wy = fx - c0, fy - r0
        h = self._terrain_h
        return (h[r0, c0] * (1 - wx) * (1 - wy) + h[r0, c0 + 1] * wx * (1 - wy)
                + h[r0 + 1, c0] * (1 - wx) * wy + h[r0 + 1, c0 + 1] * wx * wy)

    def _local_heightmap(self, qpos) -> np.ndarray:
        """Ported verbatim from envs/go1_env.py's own _local_heightmap: terrain height (m) at a
        3x3 grid of points ahead of the trunk, in the trunk's OWN forward/lateral frame (rotated
        by current yaw), relative to the height directly under the trunk."""
        x0, y0 = qpos[0], qpos[1]
        qw, qx, qy, qz = qpos[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        c, s = np.cos(yaw), np.sin(yaw)
        fwd, lat = np.array([0.15, 0.35, 0.55]), np.array([-0.15, 0.0, 0.15])
        F, L = np.meshgrid(fwd, lat, indexing="ij")
        wx = x0 + F * c - L * s
        wy = y0 + F * s + L * c
        h = self._terrain_height(wx.ravel(), wy.ravel()) - self._terrain_height(x0, y0)
        return h.astype(np.float32)

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
        parts = [v_body, w_body, [roll, pitch], qpos12, qvel12, self._last_tau_cmd, cmd]
        if self.use_terrain_heightmap:
            parts.append(self._local_heightmap(data.qpos))
        obs = np.concatenate(parts)
        return obs.astype(np.float32)

    def step(self, action):
        residual = np.clip(action, -1.0, 1.0) * self.residual_scale

        time_now_s = float(self.mujoco_go2.data.time)
        self.mujoco_go2.update_pin_with_mujoco(self.go2)

        qp_failed = False
        if (self._ctrl_i % STEPS_PER_MPC) == 0:
            self.traj.generate_traj(self.go2, self.gait, time_now_s, self.target_speed,
                                     self.target_lateral, STAND_HEIGHT, self.target_yaw_rate,
                                     time_step=MPC_DT)
            try:
                sol = self.mpc.solve_QP(self.go2, self.traj, False)
                N = self.traj.N
                w_opt = sol["x"].full().flatten()
                self._U_opt = w_opt[12 * N:].reshape((12, N), order="F")
            except RuntimeError:
                # CasADi/OSQP raise a plain RuntimeError ("conic process failed") when the QP is
                # infeasible or numerically ill-conditioned given the robot's current state --
                # genuinely a Python exception, but one this env must catch itself: confirmed by
                # direct reproduction that letting it propagate up through SubprocVecEnv kills the
                # whole worker process (EOFError on the parent's recv -- not a segfault, just an
                # uncaught exception inside the worker's own step-handling loop). Keep the last
                # successful solution for this one tick (a reasonable one-tick-stale fallback) and
                # end the episode -- a state the MPC itself can't solve for is exactly the kind of
                # thing this episode shouldn't continue from.
                qp_failed = True

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
        # A QP solver failure gets the same penalty as a fall (not a separate, milder case) --
        # a state the base controller itself can't find a feasible solution for is at least as
        # bad as falling over, and the policy should learn to avoid causing either one.
        reward = -5.0 if qp_failed else (1.0 + r_speed + r_lateral + r_yaw_rate + r_effort)

        fell = base_height < FALL_HEIGHT or abs(roll) > FALL_TILT or abs(pitch) > FALL_TILT
        terminated = bool(fell or qp_failed)
        truncated = bool(self._ctrl_i >= self.ctrl_steps)

        info = {
            "base_height": base_height, "roll": roll, "pitch": pitch, "qp_failed": qp_failed,
            "v_body": v_body.copy(), "w_body": w_body.copy(),
            # Same key ("reward_components") and flat-dict-of-floats shape train.py's own
            # RewardComponentLoggingCallback already expects, so it's reusable unmodified here.
            "reward_components": {
                "speed": r_speed, "lateral": r_lateral, "yaw_rate": r_yaw_rate, "effort": r_effort,
            },
        }
        return obs, float(reward), terminated, truncated, info
