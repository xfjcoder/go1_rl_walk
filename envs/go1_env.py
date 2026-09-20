"""
Go1FlatEnv: a Gymnasium environment that trains a Unitree Go1 to walk
forward on flat terrain using MuJoCo.

This is Stage 1 of a terrain curriculum. The env is written so later
stages (rough terrain, slopes, stairs) only need to:
  1. swap/extend the MJCF terrain geometry, and
  2. optionally widen the domain-randomization ranges below,
without touching the reward/observation logic.

Usage:
    from envs.go1_env import Go1FlatEnv
    env = Go1FlatEnv(render_mode="human")
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DEFAULT_XML = os.path.join(ASSET_DIR, "go1.xml")

JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ACTUATOR_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
FOOT_TOUCH_SENSORS = ["FR_touch", "FL_touch", "RR_touch", "RL_touch"]
FOOT_SITES = ["FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"]  # same order
THIGH_IDX = np.array([1, 4, 7, 10])  # FR, FL, RR, RL thigh indices in the 12-dim joint/action arrays
CALF_IDX = np.array([2, 5, 8, 11])   # FR, FL, RR, RL calf indices
GAIT_PHASE_OFFSETS = {
    # FR, FL, RR, RL. "trot": diagonal pairs together (FR+RL, FL+RR) -- what we've been
    # forcing all along. "bound": front pair together, rear pair together, alternating --
    # tried after repeatedly observing the policy seem to strain toward a gallop-like
    # coordination even while the diagonal pattern was being mechanically enforced,
    # suggesting this robot's dynamics may make a bound more natural than a trot.
    "trot": np.array([0.0, 0.5, 0.5, 0.0]),
    "bound": np.array([0.0, 0.0, 0.5, 0.5]),
}

# Standing/default joint pose (matches the "stand" keyframe in go1.xml)
# Standing/default joint pose (matches the "stand" keyframe in go1.xml). Same
# numeric values for every leg, same bend DIRECTION for front and rear (no
# geometric mirroring) per explicit preference -- an earlier version applied a
# 180-degree rotation to the rear thigh bodies so front/rear legs bent in
# mirrored directions (matching real quadruped anatomy, where front "elbow"
# bends back and rear "knee" bends forward), which gave good load-bearing
# symmetry but an asymmetric visual. That rotation has been removed.
#
# thigh=0.6, calf=-1.2 gives a moderate ~111-degree knee bend (vs the original
# 0.9/-1.8's ~77 degrees) -- less-severe bending should also reduce whatever
# caused the front/rear height asymmetry observed with the original, steeper
# angles. Verify the actual standing pose with smoke_test.py --record; if
# there's still a front/rear height difference, front and rear may need
# slightly different (but same-direction) numeric defaults to compensate.
DEFAULT_JOINT_POS = np.array([0.0, 0.6, -1.2] * 4, dtype=np.float32)


class Go1FlatEnv(gym.Env):
    """Flat-terrain forward-walking task for the Go1 quadruped."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        render_mode: str | None = None,
        target_speed: float = 1.0,      # m/s, forward walking speed to track
        control_hz: int = 50,           # policy control frequency
        max_episode_seconds: float = 20.0,
        action_scale: float = 0.5,      # scales policy output before adding to default pose
        domain_randomize: bool = True,  # light randomization even on flat terrain (recommended)
        trot_symmetry_weight: float = 0.15,  # bonus for diagonal-pair (FR+RL / FL+RR) trot pattern;
                                              # set to 0.0 to let any gait emerge unconstrained
        foot_clearance_weight: float = 0.08,  # bonus for lifting a swinging foot toward target_clearance;
                                               # without this, tiny/fast low-clearance shuffling can score
                                               # just as well as a bold, visible stride
        target_clearance: float = 0.04,       # target foot lift height (m) during swing phase
        heading_weight: float = 0.5,          # penalize yaw deviation from straight-ahead
        lateral_position_weight: float = 0.3,  # penalize y-position drift from the start line
        max_foot_duty_cycle: float = 0.75,    # a foot averaging more ground-contact time than
                                               # this gets penalized, regardless of other gait
                                               # terms -- directly prevents a foot from just
                                               # never participating (e.g. permanently-dragging
                                               # rear legs while only front legs step)
        min_foot_duty_cycle: float = 0.2,     # a foot averaging LESS ground-contact time than
                                               # this is ALSO penalized. Without this, a foot can
                                               # "opt out" the opposite way -- staying lifted
                                               # permanently, never touching down, which is just
                                               # as dysfunctional as constant dragging but wasn't
                                               # penalized at all before (confirmed: one leg stayed
                                               # lifted "all the time" while others dragged).
        foot_duty_weight: float = 0.3,
        ground_height_threshold: float = 0.015,  # a foot below this height counts as "grounded"
                                                  # for ALL gait-timing reward terms (gait,
                                                  # trot_symmetry, foot_duty, phase_match). Based
                                                  # on height rather than the raw contact-force
                                                  # sensor, whose tiny (1e-3) force threshold let
                                                  # a foot register as "swinging" from a momentary
                                                  # force dip with no real height change.
        gait_period: float = 0.7,             # seconds per full stride cycle (2 diagonal beats).
                                               # Only used if phase_match_weight > 0 or
                                               # use_gait_reference is True -- both default off now.
        gait_style: str = "trot",             # "trot" (diagonal pairs: FR+RL, FL+RR) or "bound"
                                               # (front pair + rear pair alternating). Controls both
                                               # the mechanical thigh reference's timing AND the
                                               # phase_match/trot_symmetry reward targets. Try "bound"
                                               # if the policy seems to be straining toward a
                                               # gallop-like coordination despite trot being enforced
                                               # -- this robot's dynamics may simply make a bound more
                                               # natural than a trot.
        phase_match_weight: float = 0.0,      # reward for matching a PRESCRIBED diagonal-trot
                                               # timing against a fixed external clock. Defaulted to
                                               # 0.0 -- across many runs, hand-picked clock parameters
                                               # (period, amplitude) either partially worked or made
                                               # things worse depending on the exact guess, because
                                               # they impose a rhythm from outside rather than letting
                                               # the policy find one consistent with the robot's own
                                               # dynamics. See air_time_weight below for the
                                               # alternative (proven in the literature: Rudin et al.,
                                               # "Learning to Walk in Minutes", ETH Zurich 2022) which
                                               # shapes rhythm without imposing a fixed clock.
        air_time_weight: float = 1.0,         # reward, on touchdown, for how long that foot was
                                               # airborne relative to target_air_time. Unlike
                                               # phase_match, this doesn't impose WHEN a foot should
                                               # swing -- only that swings, once taken, last a
                                               # reasonable duration. Lets the policy discover its
                                               # own step frequency rather than fighting a guessed one.
        target_air_time: float = 0.2,         # seconds; touchdowns faster than this are penalized
                                               # (discourages shuffling), slower are rewarded up to
                                               # this point (encourages a real, deliberate step).
        use_gait_reference: bool = False,     # bake a prescribed thigh-swing oscillation directly
                                               # into the physics (see step()), so no leg can ever
                                               # "opt out" of stepping. Defaulted OFF now in favor of
                                               # air_time_weight: hard-forcing a rhythm reliably beat
                                               # the "some legs drag" problem, but then fought the
                                               # policy on balance for every gait_period value tried
                                               # (0.7 partial success with falls, 1.0 caused marching-
                                               # in-place with no net propulsion). Still available
                                               # (--use-gait-reference) if air_time alone proves
                                               # insufficient to keep all four legs participating.
        use_calf_reference: bool = False,     # ALSO force a prescribed calf-flexion oscillation.
                                               # Kept separate from use_gait_reference (which now
                                               # only covers thigh) and defaulted OFF: a first
                                               # attempt at this caused a catastrophic regression
                                               # (ep_len_mean ~29/1000, robot buckling/kneeling
                                               # from the very first checkpoint), most likely from
                                               # an incorrect sign guess on which direction of calf
                                               # motion lifts the foot -- verify visually via
                                               # smoke_test.py --record before re-enabling this for
                                               # real training.
        gait_swing_amplitude: float = 0.35,   # radians of prescribed thigh oscillation
        thigh_residual_scale: float = 0.15,   # policy's residual authority over thigh target,
                                               # rad -- kept well below gait_swing_amplitude so
                                               # the reference always dominates and can't be
                                               # cancelled out by a learned "stay still" residual
        calf_lift_amplitude: float = 0.3,     # rad of prescribed extra calf flexion during the
                                               # swing half of the cycle, to lift the foot for
                                               # clearance -- completes the leg reference so a
                                               # foot can't "opt out" by keeping calf fixed at one
                                               # extreme (dragging: always extended; dangling:
                                               # always flexed) regardless of the forced thigh
                                               # motion, which is exactly what happened when only
                                               # thigh was constrained. NOTE: the sign here (more-
                                               # negative calf = more flexed = higher clearance) is
                                               # inferred from the joint range/default, not
                                               # empirically verified (no MuJoCo in this sandbox).
                                               # If clearance ends up happening on the wrong half
                                               # of the cycle, flip this value's sign and retrain.
        calf_residual_scale: float = 0.15,    # policy's residual authority over calf target, rad
        kp: float = 40.0,                      # PD position gain, N*m/rad. A standalone scripted-
                                               # walking test confirmed 40.0 gives poor OPEN-LOOP
                                               # tracking under load (~20 degree error) and 80.0
                                               # fixes that -- but transferring kp=80 directly to
                                               # RL was a mistake: it caused catastrophic tumbling
                                               # (ang_vel roughly 10x worse than any prior run).
                                               # A stiffer gain amplifies the consequences of a
                                               # still-learning policy's noisy/imperfect actions
                                               # into violent corrective torque, unlike a scripted
                                               # trajectory which has no such noise to amplify.
                                               # Back to the original safe value for RL.
        kd: float = 1.0,                      # PD velocity gain, N*m*s/rad
        camera: str = "track",          # which named camera to use ("track", "chase_rear", "topdown")
    ):
        super().__init__()
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.target_speed = target_speed
        self.action_scale = action_scale
        self.domain_randomize = domain_randomize
        self.trot_symmetry_weight = trot_symmetry_weight
        self.foot_clearance_weight = foot_clearance_weight
        self.target_clearance = target_clearance
        self.heading_weight = heading_weight
        self.lateral_position_weight = lateral_position_weight
        self.max_foot_duty_cycle = max_foot_duty_cycle
        self.min_foot_duty_cycle = min_foot_duty_cycle
        self.foot_duty_weight = foot_duty_weight
        self.ground_height_threshold = ground_height_threshold
        self.gait_period = gait_period
        self.phase_match_weight = phase_match_weight
        self.air_time_weight = air_time_weight
        self.target_air_time = target_air_time
        self.use_gait_reference = use_gait_reference
        self.gait_style = gait_style
        self.use_calf_reference = use_calf_reference
        self.gait_swing_amplitude = gait_swing_amplitude
        self.thigh_residual_scale = thigh_residual_scale
        self.calf_lift_amplitude = calf_lift_amplitude
        self.calf_residual_scale = calf_residual_scale
        self._kp_value = kp
        self._kd_value = kd
        self.camera = camera

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.sim_dt = self.model.opt.timestep
        self.control_hz = control_hz
        self.n_substeps = max(1, int(round(1.0 / (control_hz * self.sim_dt))))
        self.max_episode_steps = int(max_episode_seconds * control_hz)

        self._joint_qpos_adr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JOINT_NAMES]
        )
        self._joint_qvel_adr = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JOINT_NAMES]
        )
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES]
        )
        self._foot_site_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
        )
        self._touch_sensor_adr = {
            n: self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)]
            for n in FOOT_TOUCH_SENSORS
        }
        self._imu_gyro_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angular_velocity")
        ]
        self._imu_quat_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        ]
        self._imu_vel_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "trunk_linvel")
        ]

        joint_range = self.model.jnt_range[
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        ]
        torque_range = self.model.actuator_ctrlrange[self._actuator_ids]

        # Action = target joint position offsets, PD-tracked. This is the
        # standard approach for sim-to-real quadruped locomotion (Rudin et
        # al. 2022 style) and is far more sample-efficient than raw torque.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self._joint_range = joint_range
        self._torque_range = torque_range
        self._kp = self._kp_value   # PD position gain (N*m/rad)
        self._kd = self._kd_value   # PD velocity gain (N*m*s/rad)

        obs_dim = 3 + 3 + 4 + 12 + 12 + 12 + 3 + 2  # see _get_obs for layout (+2 for gait-phase clock)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._prev_action = np.zeros(12, dtype=np.float32)
        self._step_count = 0
        self._foot_contact_ema = np.zeros(4, dtype=np.float32)  # per-foot running contact fraction
        self._foot_air_time = np.zeros(4, dtype=np.float32)     # time since each foot last touched down
        self._prev_touches = np.array([True, True, True, True])  # avoids a spurious touchdown event
                                                                    # on the very first step after reset
        self._rng = np.random.default_rng()

        self._viewer = None
        if render_mode == "human":
            from mujoco import viewer as mj_viewer
            self._viewer = mj_viewer.launch_passive(self.model, self.data)
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._viewer.cam.fixedcamid = cam_id

    # ------------------------------------------------------------------ #
    # Core Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        if self.domain_randomize:
            # Small pose/velocity jitter so the policy doesn't overfit to
            # one exact starting state. Keep this light on flat terrain;
            # widen substantially once you move to rough terrain stages.
            self.data.qpos[self._joint_qpos_adr] += self._rng.uniform(-0.05, 0.05, 12)
            self.data.qpos[2] += self._rng.uniform(-0.01, 0.01)
            yaw = self._rng.uniform(-0.1, 0.1)
            self.data.qpos[3:7] = self._quat_from_yaw(yaw)
            # Randomize floor friction slightly (helps sim-to-real transfer)
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            self.model.geom_friction[floor_id, 0] = self._rng.uniform(0.6, 1.1)

        mujoco.mj_forward(self.model, self.data)
        self._prev_action[:] = 0.0
        self._step_count = 0
        self._foot_contact_ema[:] = 0.0
        self._foot_air_time[:] = 0.0
        self._prev_touches[:] = True

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_qpos = DEFAULT_JOINT_POS + self.action_scale * action

        if self.use_gait_reference:
            # Force the thigh joints to follow a prescribed diagonal-trot oscillation.
            # This is a mechanical guarantee, not a reward incentive -- across many
            # training runs, reward shaping alone (contact-count bonuses, duty-cycle
            # penalties, explicit phase-timing rewards, all combined with better
            # exploration) still repeatedly let some legs "opt out" of stepping
            # entirely, since PPO could always find some way to satisfy the reward
            # without actually using every leg. Baking the swing motion into the
            # physics itself removes that option: the policy's action for these 4
            # indices only contributes a small residual (thigh_residual_scale, kept
            # well below gait_swing_amplitude), it can no longer cancel the swing out.
            phase = self._gait_phase()
            leg_phases = (phase + GAIT_PHASE_OFFSETS[self.gait_style]) % 1.0
            thigh_reference = DEFAULT_JOINT_POS[THIGH_IDX] + \
                self.gait_swing_amplitude * np.sin(2 * np.pi * leg_phases)
            target_qpos[THIGH_IDX] = thigh_reference + self.thigh_residual_scale * action[THIGH_IDX]

            if self.use_calf_reference:
                # Calf reference: flex more (more negative, per the joint's range) during the
                # same half-cycle where thigh swings through its sin peak, lifting the foot for
                # clearance; relax back toward the default stance extension the other half.
                calf_lift = self.calf_lift_amplitude * np.maximum(0.0, np.sin(2 * np.pi * leg_phases))
                calf_reference = DEFAULT_JOINT_POS[CALF_IDX] - calf_lift
                target_qpos[CALF_IDX] = calf_reference + self.calf_residual_scale * action[CALF_IDX]

        target_qpos = np.clip(target_qpos, self._joint_range[:, 0], self._joint_range[:, 1])

        for _ in range(self.n_substeps):
            q = self.data.qpos[self._joint_qpos_adr]
            dq = self.data.qvel[self._joint_qvel_adr]
            torque = self._kp * (target_qpos - q) - self._kd * dq
            torque = np.clip(torque, self._torque_range[:, 0], self._torque_range[:, 1])
            self.data.ctrl[self._actuator_ids] = torque
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated = self._check_termination()
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        self._prev_action = action.copy()

        if self._viewer is not None:
            self._viewer.sync()

        info = {"reward_components": reward_info}
        return obs, reward, terminated, truncated, info

    def render(self, camera: str | None = None):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data, camera=camera or self.camera)
            return renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    def _gait_phase(self) -> float:
        """Where we are in the prescribed stride cycle, in [0, 1)."""
        episode_time = self._step_count / self.control_hz
        return (episode_time % self.gait_period) / self.gait_period

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        quat = self.data.sensordata[self._imu_quat_adr:self._imu_quat_adr + 4]
        gravity_vec = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        ang_vel = self.data.sensordata[self._imu_gyro_adr:self._imu_gyro_adr + 3]

        joint_pos = self.data.qpos[self._joint_qpos_adr] - DEFAULT_JOINT_POS
        joint_vel = self.data.qvel[self._joint_qvel_adr]

        command = np.array([self.target_speed, 0.0, 0.0], dtype=np.float32)  # vx, vy, yaw_rate

        phase = self._gait_phase()
        phase_clock = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32)

        # Layout (dim 51): gravity(3) + ang_vel(3) + base_quat(4) + joint_pos(12)
        #                   + joint_vel(12) + prev_action(12) + command(3) + phase_clock(2)
        obs = np.concatenate([
            gravity_vec, ang_vel, quat, joint_pos, joint_vel, self._prev_action, command, phase_clock
        ]).astype(np.float32)
        return obs


    # ------------------------------------------------------------------ #
    # Reward: standard quadruped locomotion shaping (Rudin/Hwangbo-style)
    # ------------------------------------------------------------------ #
    def _compute_reward(self, action):
        quat = self.data.sensordata[self._imu_quat_adr:self._imu_quat_adr + 4]
        lin_vel_world = self.data.sensordata[self._imu_vel_adr:self._imu_vel_adr + 3]
        ang_vel = self.data.sensordata[self._imu_gyro_adr:self._imu_gyro_adr + 3]
        gravity_vec = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))

        # 1. Track forward (x) velocity toward target_speed. Error is normalized
        # by target_speed rather than used as an absolute value -- otherwise a low
        # target_speed makes standing still (small absolute error) score almost as
        # well as actually hitting the target, which is exactly what happened at
        # target_speed=0.2: standing still scored 0.923/1.0, removing most of the
        # incentive to move at all. Normalizing keeps "standing still" penalized
        # to the same relative degree (~0.135) regardless of target_speed.
        vel_error = self.target_speed - lin_vel_world[0]
        normalized_error = vel_error / max(self.target_speed, 0.1)
        r_velocity = np.exp(-2.0 * normalized_error ** 2)

        # 2. Penalize lateral / vertical VELOCITY drift
        r_lateral = -0.5 * (lin_vel_world[1] ** 2 + lin_vel_world[2] ** 2)

        # 2b. Penalize lateral / heading POSITION drift directly. The velocity term
        # above only discourages instantaneous sideways speed -- a small, constant
        # yaw rate costs almost nothing per step under ang_vel below, but compounds
        # into significant heading/position drift over a 20s episode (this is what
        # produced "moves right instead of straight" even though nothing looked
        # obviously wrong at the per-step level). Track actual y-position and yaw
        # deviation from the start pose directly so drift is corrected, not just
        # discouraged in the instant.
        y_pos = self.data.qpos[1]
        qw, qx, qy, qz = quat
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        r_heading = -self.heading_weight * (yaw ** 2) - self.lateral_position_weight * (y_pos ** 2)

        # 3. Penalize roll/pitch (staying upright): gravity_vec should be ~[0,0,-1].
        # NOTE: this term is blind to yaw -- pure yaw rotation doesn't change the
        # body-frame gravity vector at all, which is exactly why heading drift went
        # uncorrected before r_heading was added above.
        r_orientation = -1.0 * (gravity_vec[0] ** 2 + gravity_vec[1] ** 2)

        # 4. Penalize excessive base angular velocity (yaw/roll/pitch rate)
        r_ang_vel = -0.05 * np.sum(np.square(ang_vel))

        # 5. Torque / energy cost
        torque = self.data.ctrl[self._actuator_ids]
        r_torque = -0.0002 * np.sum(np.square(torque))

        # 6. Action smoothness (penalize jerky policy outputs). Raised from the
        # original 0.01 -- a weak penalty here lets the policy satisfy the foot-
        # contact-count and trot-symmetry terms with tiny, high-frequency corrections
        # instead of a bold, visible stride, which is what produced the "small fast
        # shuffling" gait once the policy fully converged.
        r_action_rate = -0.03 * np.sum(np.square(action - self._prev_action))

        # 7. Foot air-time bonus: encourage lifting feet (discourages shuffling/dragging)
        #
        # "touches" is defined by actual foot HEIGHT (ground_height_threshold), not the raw
        # contact-force sensor. The force sensor's threshold (1e-3, a tiny force) let a foot
        # register as "swinging" from a momentary, imperceptible force dip with no real height
        # change -- exactly the loophole that let rear legs satisfy the duty-cycle penalty while
        # still visually dragging (confirmed: front legs showed genuine large-amplitude stepping
        # while rear legs kept the same "quick tiny/dragging" pattern despite the penalty).
        # Requiring a real height clearance closes this for every term that uses "touches" below
        # (gait, trot_symmetry, foot_duty, phase_match all share this one definition).
        foot_heights_for_contact = self.data.site_xpos[self._foot_site_ids, 2]
        touches = foot_heights_for_contact < self.ground_height_threshold
        n_contacts = touches.sum()

        # 6b. Feet air-time reward (Rudin et al. 2022 "Learning to Walk in Minutes" style):
        # on the step a foot touches DOWN, reward it for how long it had been airborne,
        # relative to target_air_time. Unlike phase_match/gait_period, this never specifies
        # WHEN a foot should swing or for how long exactly -- only that a completed swing
        # was a reasonable duration (discourages both tiny fast shuffling and a foot stuck
        # up too long). The policy is free to find its own step frequency and timing, rather
        # than matching a fixed clock we picked without empirical grounding.
        touchdown = touches & (~self._prev_touches)
        r_air_time = self.air_time_weight * float(
            np.sum(touchdown.astype(np.float32) * (self._foot_air_time - self.target_air_time))
        )
        dt_control = 1.0 / self.control_hz
        self._foot_air_time = np.where(touches, 0.0, self._foot_air_time + dt_control)
        self._prev_touches = touches.copy()

        # A canonical trot has exactly 2 feet down. Previously this rewarded ANY
        # count from 1-3 equally, which let a persistent 3-down/1-stepping pattern
        # (rear legs always planted, only front legs doing anything) score exactly
        # as well as a real 2-2 trot -- removing any pressure to actually use all
        # four legs. Now only a genuine 2-2 split gets the bonus; 1 or 3 is neutral,
        # 0 or 4 (stumble/frozen) is still penalized.
        if n_contacts == 2:
            r_gait = 0.08
        elif n_contacts == 3:
            r_gait = 0.0   # slightly sloppy (an extra foot down) but not dangerous
        elif n_contacts == 1:
            # Single-point-of-support: dramatically less stable than either a proper
            # 2-2 trot or a sloppy 3-down stance (no roll stability at all -- the
            # robot can rock and fall either way). Previously treated identically to
            # n_contacts==3 (both scored 0.0, neutral), which gave no specific
            # incentive to avoid this far more dangerous case. This was observed
            # directly: the robot took a couple of good steps, then ended up
            # supported on only the front-right foot while all three other legs
            # were simultaneously airborne (diagonal-pair synchrony broken down),
            # and fell from exactly that configuration.
            r_gait = -0.15
        else:  # 0 or 4
            r_gait = -0.05

        # 7a. Per-foot duty-cycle penalty: track each foot's running fraction of
        # time spent in contact, and penalize any foot whose average exceeds
        # max_foot_duty_cycle. This is what actually stops a foot from "opting
        # out" of the gait entirely (e.g. permanently-dragging rear legs) --
        # unlike the terms above, this evaluates EVERY foot every step
        # regardless of whether it's currently swinging, so there's no way for
        # a foot to just never be checked.
        ema_decay = 0.98  # ~50-step (1s) time constant at 50Hz control
        self._foot_contact_ema = ema_decay * self._foot_contact_ema + (1 - ema_decay) * touches.astype(np.float32)
        duty_excess_high = np.maximum(self._foot_contact_ema - self.max_foot_duty_cycle, 0.0)
        duty_excess_low = np.maximum(self.min_foot_duty_cycle - self._foot_contact_ema, 0.0)
        r_foot_duty = -self.foot_duty_weight * float(np.sum(duty_excess_high) + np.sum(duty_excess_low))

        # 7a2. Gait-phase tracking: reward matching a PRESCRIBED footfall timing (trot: diagonal
        # pairs FR+RL / FL+RR; bound: front pair / rear pair), rather than any loose "some
        # acceptable split" check. This gives the policy an explicit, unambiguous reference to
        # track -- much easier for per-step Gaussian exploration to find than "discover a good
        # rhythm from scratch."
        phase = self._gait_phase()
        if self.gait_style == "bound":
            front_stance = phase < 0.5
            desired_stance = np.array([front_stance, front_stance, not front_stance, not front_stance])  # FR,FL,RR,RL
        else:  # "trot"
            fr_rl_stance = phase < 0.5
            desired_stance = np.array([fr_rl_stance, not fr_rl_stance, not fr_rl_stance, fr_rl_stance])  # FR,FL,RR,RL
        r_phase_match = self.phase_match_weight * float(np.mean(touches == desired_stance))

        # 7b. Footfall-pattern symmetry bonus (trot: diagonal pairs; bound: front/rear pairs).
        # touches is ordered [FR, FL, RR, RL]. Without this term the reward above only cares
        # about *how many* feet are down, not *which* ones -- which lets the policy converge
        # on lopsided, asymmetric gaits (e.g. one leg taking large strides while the others
        # shuffle) that still score well. This term nudges (not forces) it toward the target
        # footfall pattern.
        fr, fl, rr, rl = touches
        if self.gait_style == "bound":
            pair_match = float(fr == fl) + float(rr == rl)       # front pair, rear pair agree
            pairs_alternate = float(fr != rr)                     # front pair differs from rear pair
        else:  # "trot"
            pair_match = float(fr == rl) + float(fl == rr)       # diagonal pairs agree
            pairs_alternate = float(fr != fl)                     # the two diagonals differ
        # Only award this bonus for a genuine 2-2 split (n_contacts == 2). Without
        # this guard, a static all-feet-planted pose (n_contacts == 4, never
        # changing) trivially satisfies "pairs match" and "not equal to the other
        # pair" is moot -- it scored close to full marks for literally standing
        # still, which was part of why the policy collapsed to a frozen pose once
        # training stabilized.
        if n_contacts == 2:
            r_trot_symmetry = self.trot_symmetry_weight * (pair_match / 2.0 + pairs_alternate) / 2.0
        else:
            r_trot_symmetry = 0.0

        # 7c. Foot-clearance bonus: reward swinging (non-contact) feet for actually
        # lifting off the ground toward target_clearance. Neither the contact-count
        # term nor the trot-symmetry term above cares about *how high* a foot lifts
        # during its swing phase -- only whether it's touching or not -- so without
        # this, tiny near-ground shuffling motions satisfy those terms just as well
        # as a bold, visible stride, and the policy has no incentive to prefer the
        # latter once it's found the former.
        foot_heights = foot_heights_for_contact  # same values, already computed above
        swing = ~touches
        if swing.any():
            clearance_frac = np.minimum(foot_heights[swing], self.target_clearance) / self.target_clearance
            r_foot_clearance = self.foot_clearance_weight * float(np.mean(clearance_frac))
        else:
            r_foot_clearance = 0.0

        # 8. Survival bonus (alive each control step). Reduced from the original 0.5:
        # at 0.5/step, up to 500 total over a full episode, this alone was likely large
        # enough to dominate the incentive calculus regardless of gait quality -- a
        # policy that plays it maximally safe collects nearly all of this "for free"
        # while genuine trotting risks losing it by falling. Still present so survival
        # isn't ignored, but no longer overwhelming the gait-quality-specific terms.
        r_alive = 0.2

        # 9. Height maintenance (target ~0.30-0.33m standing height)
        height = self.data.qpos[2]
        r_height = -2.0 * (height - 0.30) ** 2 if height < 0.30 else 0.0

        weights_applied = dict(
            velocity=r_velocity * 1.5,
            lateral=r_lateral,
            heading=r_heading,
            orientation=r_orientation,
            ang_vel=r_ang_vel,
            torque=r_torque,
            action_rate=r_action_rate,
            gait=r_gait,
            foot_duty=r_foot_duty,
            air_time=r_air_time,
            phase_match=r_phase_match,
            trot_symmetry=r_trot_symmetry,
            foot_clearance=r_foot_clearance,
            alive=r_alive,
            height=r_height,
        )
        total = float(sum(weights_applied.values()))
        return total, weights_applied

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #
    def _check_termination(self):
        quat = self.data.sensordata[self._imu_quat_adr:self._imu_quat_adr + 4]
        gravity_vec = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        fell_over = gravity_vec[2] > -0.5  # trunk tilted past ~60 deg from vertical
        too_low = self.data.qpos[2] < 0.15
        too_high = self.data.qpos[2] > 0.6
        return bool(fell_over or too_low or too_high)

    # ------------------------------------------------------------------ #
    # Small math helpers (no scipy dependency)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quat_rotate_inv(quat, vec):
        """Rotate `vec` from world frame into body frame given body quat (w,x,y,z)."""
        w, x, y, z = quat
        # Inverse rotation = rotation by conjugate quaternion
        qw, qx, qy, qz = w, -x, -y, -z
        # Hamilton product: q * v * q^-1, v as pure quaternion (0, vec)
        vx, vy, vz = vec
        # t = 2 * cross(q_xyz, v)
        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)
        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)
        return np.array([rx, ry, rz])

    @staticmethod
    def _quat_from_yaw(yaw):
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
