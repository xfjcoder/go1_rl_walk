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

# Rough-terrain heightfield (only injected into the model when terrain is enabled; the flat env is unchanged).
# Grid covers x in [-2.5, 22.5] m and y in [-3, 3] m at 5 cm cells, so a 20 s episode at 1 m/s stays on it.
HF_CELL = 0.05
HF_CX = 10.0          # hfield geom centre x (m)
HF_HALF_X = 12.5
HF_HALF_Y = 3.0
HF_ELEV = 6.0          # elevation_z: max terrain height (m); hfield_data in [0,1] scales to [0, HF_ELEV].
                       # Generous headroom for slopes (e.g. 20 deg over a 10 m ramp needs ~3.6 m) --
                       # doesn't cost anything for the much smaller rough-terrain bump amplitudes (<=0.12 m).
DEFAULT_RAMP_LENGTH = 8.0   # horizontal distance (m) a slope climbs/descends over, before leveling into a plateau
HF_PAD_START, HF_PAD_END = 0.5, 1.5   # x range over which terrain fades in (flat start pad before it)

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
        target_clearance: float = 0.04,       # target foot lift height (m) during swing phase; the
                                               # EFFECTIVE per-episode target is max(this, stair_h + 0.03)
                                               # -- see _episode_target_clearance -- so a tall stair riser
                                               # actually raises the reward's incentive to lift higher,
                                               # instead of capping out at 4cm regardless of the obstacle
                                               # (confirmed: this cap was why a trained policy got physically
                                               # stuck at the first stair on a 12cm riser -- median swing
                                               # height was only 6.6cm, well under what's needed to clear it,
                                               # because lifting higher than 4cm earned no extra reward).
        heading_weight: float = 0.5,          # penalize yaw deviation from straight-ahead
        lateral_position_weight: float = 0.3,  # penalize y-position drift from the start line
        body_frame_velocity: bool = False,     # track / penalize velocity in the BODY frame (forward = where
                                               # the trunk points) instead of world x/y. With world-frame
                                               # tracking a policy can veer off-axis (pilot D ended at -19 deg
                                               # yaw) and still be rewarded for world-x progress.
        terrain_amplitude_range: tuple | None = None,  # (lo, hi): enable a random heightfield; each reset samples
                                               # amplitude ~ U(lo, terrain_amp_max_current). The current max
                                               # starts at hi and can be ramped by a curriculum callback.
        terrain_amplitude: float | None = None,  # fixed amplitude (m, peak-to-peak) -- for evaluation
        slope_range: tuple | None = None,      # (lo, hi) degrees: enable a ramp -- flat pad, then a
                                               # constant-grade climb/descent (random direction each
                                               # episode) over ramp_length, then a flat plateau. Sampled
                                               # like terrain_amplitude_range (U(lo, slope_deg_max_current)).
        slope_deg: float | None = None,        # fixed slope (deg, signed: + uphill, - downhill) -- for evaluation
        ramp_length: float = DEFAULT_RAMP_LENGTH,  # metres of horizontal run to reach the target slope height
        stair_height_range: tuple | None = None,  # (lo, hi) metres: enable stairs -- flat pad, then num_stairs
                                               # steps of stair_depth tread x this riser height (random direction
                                               # each episode), then a flat plateau. Sampled like slope_range
                                               # (U(lo, stair_height_max_current)). Additive with slope/bumps.
        stair_height: float | None = None,     # fixed riser height (m, signed: + ascending, - descending) -- eval
        stair_depth: float = 0.25,             # tread depth (m); kept a multiple of HF_CELL (0.05m) for a crisp
                                               # riser edge -- a heightfield can only change height within one
                                               # grid cell, not a true vertical face, so a riser is a steep ramp
                                               # over one HF_CELL (5 cm), not a perfect right angle.
        num_stairs: int = 8,                   # number of steps before leveling into a plateau
        friction_range: tuple = (0.6, 1.1),    # floor / terrain sliding friction sampled each reset
        mass_scale_range: tuple | None = None,  # trunk mass (and inertia) scale sampled each reset
        push_velocity: float = 0.0,            # m/s: every 3-6 s add a random horizontal velocity kick of up
                                               # to +-this to the trunk (0 = off)
        command_speed_range: tuple | None = None,  # (lo, hi): sample target_speed ~ U(lo, hi_now) every reset.
                                               # hi_now starts at hi and can be ramped by a curriculum callback
                                               # via the speed_max_current attribute. None = fixed target_speed.
        gait_period_fast: float | None = None,  # if set, the trot clock period shrinks linearly from gait_period
                                               # (at 0.3 m/s) to this value (at 1.0 m/s), clipped outside, so faster
                                               # commands get a faster step rate. None = fixed gait_period.
        lateral_tracking_weight: float = 0.0,  # reward exp(-(v_side/sigma)^2) for zero sideways velocity (body frame
                                               # if body_frame_velocity). Fixes crabbing: h_clock drifted 4.6 cm/s
                                               # sideways while heading stayed straight.
        lateral_tracking_sigma: float = 0.1,
        air_time_cap: bool = False,            # cap the touchdown air-time credit at target_air_time
                                               # (credit = min(air, target) - target <= 0). Uncapped, one long
                                               # 457 ms lift of a single leg out-earned several normal steps
                                               # (run e_long_d: RR hovered, front feet shuffled 1-2 cm high).
        yaw_rate_weight: float = 0.0,          # penalty weight on yaw rate^2 (turning); ang_vel term already
                                               # covers roll/pitch/yaw rates weakly (0.05)
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
        contact_force_threshold: float = 1.0,    # N. A foot counts as "grounded" (for ALL gait-timing
                                                  # reward terms: gait, trot_symmetry, foot_duty,
                                                  # phase_match, air_time) when it has a MuJoCo contact
                                                  # with normal force above this. This replaced a foot
                                                  # HEIGHT threshold of 0.015 m, which was below the
                                                  # foot site's resting height (0.022 m = sphere radius,
                                                  # the site sits at the sphere centre), so a robot
                                                  # standing on all four feet was seen as having ZERO
                                                  # feet down and every contact-based term was wrong.
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
        self._episode_target_clearance = target_clearance
        self.heading_weight = heading_weight
        self.body_frame_velocity = body_frame_velocity
        self.yaw_rate_weight = yaw_rate_weight
        self.air_time_cap = air_time_cap
        self.command_speed_range = tuple(command_speed_range) if command_speed_range else None
        self.terrain_amplitude_range = tuple(terrain_amplitude_range) if terrain_amplitude_range else None
        self.terrain_amplitude = terrain_amplitude
        self.terrain_amp_max_current = self.terrain_amplitude_range[1] if self.terrain_amplitude_range else None
        self.slope_range = tuple(slope_range) if slope_range else None
        self.slope_deg = slope_deg
        self.slope_deg_max_current = self.slope_range[1] if self.slope_range else None
        self.ramp_length = ramp_length
        self.stair_height_range = tuple(stair_height_range) if stair_height_range else None
        self.stair_height = stair_height
        self.stair_height_max_current = self.stair_height_range[1] if self.stair_height_range else None
        self.stair_depth = stair_depth
        self.num_stairs = num_stairs
        self.terrain_enabled = (self.terrain_amplitude_range is not None or terrain_amplitude is not None
                                or self.slope_range is not None or slope_deg is not None
                                or self.stair_height_range is not None or stair_height is not None)
        self.friction_range = tuple(friction_range)
        self.mass_scale_range = tuple(mass_scale_range) if mass_scale_range else None
        self.push_velocity = push_velocity
        self.speed_max_current = self.command_speed_range[1] if self.command_speed_range else None
        self.gait_period_fast = gait_period_fast
        self.lateral_tracking_weight = lateral_tracking_weight
        self.lateral_tracking_sigma = lateral_tracking_sigma
        self._episode_gait_period = gait_period
        self.lateral_position_weight = lateral_position_weight
        self.max_foot_duty_cycle = max_foot_duty_cycle
        self.min_foot_duty_cycle = min_foot_duty_cycle
        self.foot_duty_weight = foot_duty_weight
        self.contact_force_threshold = contact_force_threshold
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

        if self.terrain_enabled:
            self.model = mujoco.MjModel.from_xml_string(self._terrain_xml(xml_path))
        else:
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
        self._foot_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n.replace("_site", ""))
             for n in FOOT_SITES]
        )
        self._foot_radius = float(self.model.geom_size[self._foot_geom_ids[0], 0])  # site is at sphere centre
        self._contact_force = np.zeros(6)
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

        # terrain / randomization state
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._trunk_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self._nominal_trunk_mass = float(self.model.body_mass[self._trunk_body])
        self._nominal_trunk_inertia = self.model.body_inertia[self._trunk_body].copy()
        self._terrain_h = None                    # (nrow, ncol) heights in metres, None on flat ground
        self._terrain_geom = None
        if self.terrain_enabled:
            self._terrain_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
            hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
            self._hf_nrow, self._hf_ncol = int(self.model.hfield_nrow[hid]), int(self.model.hfield_ncol[hid])
            self._hf_adr = int(self.model.hfield_adr[hid])
            self._hf_x0, self._hf_y0 = HF_CX - HF_HALF_X, -HF_HALF_Y
            self._terrain_h = np.zeros((self._hf_nrow, self._hf_ncol))
        self._next_push_step = 10**9

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

        if self.command_speed_range is not None:
            lo = self.command_speed_range[0]
            self.target_speed = float(self._rng.uniform(lo, max(self.speed_max_current, lo)))
        self._episode_gait_period = self._period_for_speed(self.target_speed)

        if self.mass_scale_range is not None:
            sc = float(self._rng.uniform(*self.mass_scale_range))
            self.model.body_mass[self._trunk_body] = self._nominal_trunk_mass * sc
            self.model.body_inertia[self._trunk_body] = self._nominal_trunk_inertia * sc
            mujoco.mj_setConst(self.model, self.data)
        if self.terrain_enabled:
            if self.terrain_amplitude is not None:
                amp = self.terrain_amplitude
            elif self.terrain_amplitude_range is not None:
                amp = float(self._rng.uniform(self.terrain_amplitude_range[0],
                                              max(self.terrain_amp_max_current, self.terrain_amplitude_range[0])))
            else:
                amp = 0.0
            if self.slope_deg is not None:
                slope_deg, uphill = abs(self.slope_deg), self.slope_deg >= 0
            elif self.slope_range is not None:
                slope_deg = float(self._rng.uniform(self.slope_range[0],
                                                    max(self.slope_deg_max_current, self.slope_range[0])))
                uphill = bool(self._rng.integers(0, 2))
            else:
                slope_deg, uphill = 0.0, True
            if self.stair_height is not None:
                stair_h, ascending = abs(self.stair_height), self.stair_height >= 0
            elif self.stair_height_range is not None:
                stair_h = float(self._rng.uniform(self.stair_height_range[0],
                                                  max(self.stair_height_max_current, self.stair_height_range[0])))
                ascending = bool(self._rng.integers(0, 2))
            else:
                stair_h, ascending = 0.0, True
            self._generate_terrain(amp, slope_deg, uphill, stair_h, ascending)
            self._episode_target_clearance = max(self.target_clearance, stair_h + 0.03)
        self._next_push_step = int(self._rng.integers(150, 300)) if self.push_velocity > 0 else 10**9

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
            mu = self._rng.uniform(*self.friction_range)
            self.model.geom_friction[self._floor_id, 0] = mu
            if self._terrain_geom is not None:
                self.model.geom_friction[self._terrain_geom, 0] = mu
                # The foot geoms have priority=1, so THEIR friction (1.0) overrides the floor's in every
                # foot contact -- randomizing only the floor never touched the feet. On terrain runs randomize
                # the foot friction too (flat runs keep the old behaviour so they stay reproducible).
                self.model.geom_friction[self._foot_geom_ids, 0] = mu

        mujoco.mj_forward(self.model, self.data)
        if self.terrain_enabled:
            # The stand keyframe (and its random joint jitter) leaves the foot-sphere centres BELOW the surface
            # (z = -0.002 .. -0.012 m). A plane pushes buried feet out gently, but a heightfield contact reacts
            # violently (37 contacts, ~11 kN per foot on step 0). Lift the trunk so every sole starts just above
            # the local terrain.
            foot = self.data.site_xpos[self._foot_site_ids]
            sole = foot[:, 2] - self._foot_radius - self._terrain_height(foot[:, 0], foot[:, 1])
            lift = 0.004 - float(sole.min())
            if lift > 0:
                self.data.qpos[2] += lift
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
        if self._step_count == self._next_push_step:
            self.data.qvel[0:2] += self._rng.uniform(-self.push_velocity, self.push_velocity, 2)
            self._next_push_step += int(self._rng.integers(150, 300))
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

        qw, qx, qy, qz = self.data.sensordata[self._imu_quat_adr:self._imu_quat_adr + 4]
        info = {"reward_components": reward_info,
                "base_pos": self.data.qpos[:3].copy(),
                "yaw": float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))}
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

    def _foot_contacts(self) -> np.ndarray:
        """bool[4] (FR, FL, RR, RL): foot has a contact with normal force > contact_force_threshold.
        Uses real MuJoCo contacts, so it stays correct on rough terrain (unlike a height threshold)."""
        force = np.zeros(4)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            hit = np.flatnonzero((self._foot_geom_ids == c.geom1) | (self._foot_geom_ids == c.geom2))
            if hit.size:
                mujoco.mj_contactForce(self.model, self.data, i, self._contact_force)
                force[hit[0]] += self._contact_force[0]     # a foot on a heightfield can have several contact points
        return force > self.contact_force_threshold

    # ------------------------------------------------------------------ #
    # Rough terrain (heightfield)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _terrain_xml(xml_path: str) -> str:
        """The flat model with a random-heightfield geom injected. The floor plane drops 5 cm as a safety net so
        it never coincides with the terrain surface (heights are >= 0)."""
        with open(xml_path) as f:
            xml = f.read()
        nrow, ncol = int(round(2 * HF_HALF_Y / HF_CELL)) + 1, int(round(2 * HF_HALF_X / HF_CELL)) + 1
        edits = [
            ("</asset>", f'<hfield name="terrain" nrow="{nrow}" ncol="{ncol}" '
                         f'size="{HF_HALF_X} {HF_HALF_Y} {HF_ELEV} 0.1"/>\n  </asset>'),
            ('<geom name="floor" type="plane"', '<geom name="floor" pos="0 0 -0.05" type="plane"'),
            ('<light name="sun"', f'<geom name="terrain" type="hfield" hfield="terrain" pos="{HF_CX} 0 0" '
                                  f'rgba="0.36 0.5 0.36 1" friction="0.9 0.02 0.01" condim="3"/>\n    <light name="sun"'),
        ]
        for old, new in edits:
            assert xml.count(old) == 1, f"cannot inject terrain: {old!r} not found exactly once in {xml_path}"
            xml = xml.replace(old, new)
        return xml

    def _generate_terrain(self, amp: float, slope_deg: float = 0.0, uphill: bool = True,
                          stair_h: float = 0.0, ascending: bool = True):
        """Terrain height (m), combining three independent, additive components:

        1. Random bumps: bilinearly-interpolated uniform noise in [0, amp] with a random feature size
           (15 cm bumps up to 1 m undulations), faded in over the flat start pad (HF_PAD_START..HF_PAD_END).
        2. A slope: flat at 0 up to HF_PAD_END, then a constant grade (tan(slope_deg)) over ramp_length
           metres, then a flat plateau at the resulting height for the rest of the course. `uphill=False`
           reverses which end is elevated (flat pad starts high, plateau ends at 0) rather than using a
           negative height -- a MuJoCo hfield can't represent height below its own z=0 reference plane.
        3. Stairs: flat at 0 up to HF_PAD_END, then num_stairs steps of stair_depth tread x stair_h riser,
           then a flat plateau at num_stairs*stair_h. `ascending=False` mirrors direction the same way
           `uphill` does for the slope. Each riser is a step function quantized to HF_CELL, so it's a
           steep ramp within one grid cell (5 cm), not a true vertical face -- see the ctor docstring.

        All three default to 0 (flat ground), and any combination can be nonzero at once (e.g. bumpy stairs).
        """
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
            slope = rise if uphill else rise.max() - rise   # downhill: start elevated, descend to 0

        if stair_h <= 1e-6:
            stairs = np.zeros(ncol)
        else:
            step_num = np.clip(np.floor((xw - HF_PAD_END) / self.stair_depth), 0, self.num_stairs)
            rise = stair_h * step_num
            stairs = rise if ascending else rise.max() - rise   # descending: start elevated, step down to 0

        h = np.clip(bumps + slope[None, :] + stairs[None, :], 0.0, HF_ELEV)
        self._terrain_h = h
        self.model.hfield_data[self._hf_adr:self._hf_adr + nrow * ncol] = (h / HF_ELEV).ravel()

    def _terrain_height(self, x, y):
        """Terrain surface height (m) at world (x, y); scalars or arrays. Zero on flat ground."""
        if self._terrain_h is None:
            return np.zeros(np.shape(x)) if np.ndim(x) else 0.0
        fx = np.clip((np.asarray(x) - self._hf_x0) / HF_CELL, 0.0, self._hf_ncol - 1 - 1e-9)
        fy = np.clip((np.asarray(y) - self._hf_y0) / HF_CELL, 0.0, self._hf_nrow - 1 - 1e-9)
        c0, r0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        wx, wy = fx - c0, fy - r0
        h = self._terrain_h
        return (h[r0, c0] * (1 - wx) * (1 - wy) + h[r0, c0 + 1] * wx * (1 - wy)
                + h[r0 + 1, c0] * (1 - wx) * wy + h[r0 + 1, c0 + 1] * wx * wy)

    def _period_for_speed(self, speed: float) -> float:
        if self.gait_period_fast is None:
            return self.gait_period
        f = float(np.clip((speed - 0.3) / 0.7, 0.0, 1.0))
        return (1 - f) * self.gait_period + f * self.gait_period_fast

    def _gait_phase(self) -> float:
        """Where we are in the prescribed stride cycle, in [0, 1)."""
        episode_time = self._step_count / self.control_hz
        return (episode_time % self._episode_gait_period) / self._episode_gait_period

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
        lin_vel = self._quat_rotate_inv(quat, lin_vel_world) if self.body_frame_velocity else lin_vel_world
        vel_error = self.target_speed - lin_vel[0]
        normalized_error = vel_error / max(self.target_speed, 0.1)
        r_velocity = np.exp(-2.0 * normalized_error ** 2)

        # 2. Penalize lateral / vertical VELOCITY drift
        r_lateral = -0.5 * (lin_vel[1] ** 2 + lin_vel[2] ** 2)
        r_yaw_rate = -self.yaw_rate_weight * ang_vel[2] ** 2
        r_lat_track = self.lateral_tracking_weight * float(np.exp(-(lin_vel[1] / self.lateral_tracking_sigma) ** 2))

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
        # "touches" = real foot-floor contact with normal force > contact_force_threshold (1 N),
        # shared by every term below (gait, trot_symmetry, foot_duty, phase_match, air_time).
        # A 1 N threshold ignores momentary grazes. (An earlier height-based definition used a
        # threshold below the foot's resting height, so standing feet were counted as airborne.)
        foot_xyz = self.data.site_xpos[self._foot_site_ids]
        foot_heights_for_contact = foot_xyz[:, 2] - self._foot_radius - self._terrain_height(foot_xyz[:, 0], foot_xyz[:, 1])  # sole height above the ground
        touches = self._foot_contacts()
        n_contacts = touches.sum()

        # 6b. Feet air-time reward (Rudin et al. 2022 "Learning to Walk in Minutes" style):
        # on the step a foot touches DOWN, reward it for how long it had been airborne,
        # relative to target_air_time. Unlike phase_match/gait_period, this never specifies
        # WHEN a foot should swing or for how long exactly -- only that a completed swing
        # was a reasonable duration (discourages both tiny fast shuffling and a foot stuck
        # up too long). The policy is free to find its own step frequency and timing, rather
        # than matching a fixed clock we picked without empirical grounding.
        touchdown = touches & (~self._prev_touches)
        air = np.minimum(self._foot_air_time, self.target_air_time) if self.air_time_cap else self._foot_air_time
        r_air_time = self.air_time_weight * float(
            np.sum(touchdown.astype(np.float32) * (air - self.target_air_time))
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
            tgt = self._episode_target_clearance
            clearance_frac = np.minimum(foot_heights[swing], tgt) / tgt
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
        height = self.data.qpos[2] - self._terrain_height(self.data.qpos[0], self.data.qpos[1])
        r_height = -2.0 * (height - 0.30) ** 2 if height < 0.30 else 0.0

        weights_applied = dict(
            velocity=r_velocity * 1.5,
            lateral=r_lateral,
            yaw_rate=r_yaw_rate,
            lateral_tracking=r_lat_track,
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
        trunk_h = self.data.qpos[2] - self._terrain_height(self.data.qpos[0], self.data.qpos[1])
        too_low = trunk_h < 0.15
        too_high = trunk_h > 0.6
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
