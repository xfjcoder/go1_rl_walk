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

import casadi as ca
import gymnasium as gym
import mujoco as mj
import numpy as np
import scipy.sparse as sp
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE_SRC = REPO_ROOT / "third_party" / "go2-convex-mpc" / "src"
sys.path.insert(0, str(SUBMODULE_SRC))
sys.path.insert(0, str(REPO_ROOT))

from convex_mpc.centroidal_mpc import CentroidalMPC, COST_MATRIX_Q, MU, NU, NX
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import Gait, HEIGHT_SWING
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.leg_controller import LEG_INDEX, LegController
import convex_mpc.mujoco_model as mujoco_model_module
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
import pinocchio as pin


# TerrainAwareGait._best_foothold_xy's search grid/regularization. Radius/step chosen as
# a small fraction of the leg's real reachable workspace (nowhere near kinematic limits);
# REG_WEIGHT chosen so the regularization cost at the search radius's edge is roughly the
# same order as the slope-magnitude readings seen on this project's own test slopes
# (tan(12deg)~=0.21 at the steepest previously-tested terrain) -- i.e. a candidate has to
# be meaningfully flatter than the nominal point to be worth moving toward, not just
# infinitesimally flatter.
FOOTHOLD_SEARCH_RADIUS = 0.05   # meters
FOOTHOLD_GRID_STEP = 0.025      # meters
FOOTHOLD_REG_WEIGHT = 0.15


class TerrainAwareGait(Gait):
    """Patches the submodule's own Gait (unmodified otherwise -- subclassed rather than
    edited in place, since it lives in a vendored git submodule) so footstep planning
    knows about terrain instead of hardcoding a flat-ground touchdown height.

    Root cause this fixes: `Gait.compute_touchdown_world_for_traj_purpose_only` and
    `Gait.compute_swing_traj_and_touchdown` both hardcode the touchdown point's world Z
    to a constant 0.02 (a flat-ground assumption -- see third_party/go2-convex-mpc's own
    gait.py). That constant feeds TWO places: (1) `ComTraj.generate_traj`'s predicted
    foot lever arms over the whole MPC horizon, used to build the Bd dynamics matrices
    the QP solves against -- on a slope/stair, the real moment arm's Z is wrong, biasing
    the force/torque solution over the whole horizon; (2) the actual swing-leg Cartesian
    PD target in `LegController.compute_leg_torque` -- the swing foot is commanded to
    land at flat-ground height regardless of a real slope/step under it, so it can
    contact early (confusing the MPC's own contact schedule, which still thinks that leg
    is airborne) or miss long (briefly losing a support leg the MPC assumed was already
    planted). This is a materially different fix than Stage 11's terrain_v4 experiment,
    which gave the RL RESIDUAL a terrain-heightmap observation but left MPC's own
    footstep planner blind -- here MPC's own planning is corrected directly, nothing
    downstream has to compensate for a bad plan.

    Both overridden methods call the parent (unmodified) implementation first to get
    the correct horizontal (x, y) touchdown position (never depends on terrain) and
    discard only its hardcoded z, rather than re-deriving the drift/rotation-correction
    math here -- keeps this subclass correct even if the upstream submodule's own
    horizontal footstep logic changes.

    `foothold_search`, off by default: once this class is already correcting Z at the
    nominal (x, y), the natural next step (the actual ANYmal/Spot perceptive-locomotion
    mechanism) is to also search nearby (x, y) candidates for a better foothold instead
    of blindly accepting the nominal Raibert point's horizontal location -- see
    `_best_foothold_xy`. Gated by its own flag (independent of the z-only fix above,
    which stays exactly reproducible with this off) since it's a separate, newer,
    less-tested lever.

    `swing_clearance_adaptive`, off by default: `make_swing_trajectory`'s `h_sw` (swing
    apex height) is added on top of a straight-LINE interpolation between takeoff and
    touchdown, so the actual ground clearance above the HIGHER of the two endpoints is
    `h_sw - |p0_z - pf_z| / 2`, not `h_sw` -- on flat ground the two z's are equal so
    this is moot, but ascending a stair riser (touchdown higher than takeoff) shrinks the
    true clearance over the step being climbed, the same "fixed clearance target" bug
    the pure-RL stairs track already found and only partially fixed on its own. Fixed by
    solving for the `h_sw` that keeps clearance above the higher endpoint fixed at the
    original `HEIGHT_SWING` regardless of the height difference: `HEIGHT_SWING +
    |p0_z - pf_z| / 2` (reduces to exactly `HEIGHT_SWING` when flat, i.e. bit-identical
    to before when the two feet-height endpoints match).
    """

    def __init__(self, frequency_hz, duty, terrain_height_fn, foothold_search=False,
                 swing_clearance_adaptive=False):
        super().__init__(frequency_hz, duty)
        self._terrain_height_fn = terrain_height_fn
        self._foothold_search = foothold_search
        self._swing_clearance_adaptive = swing_clearance_adaptive

    def _best_foothold_xy(self, x_nom, y_nom):
        """Searches a small grid of candidate (x, y) offsets around the nominal Raibert
        touchdown point and returns the one with the lowest cost = local terrain slope
        magnitude (a cheap proxy for "don't land on an edge/slope", same central-
        difference style as _terrain_normal) + a regularization term pulling back toward
        the nominal point (otherwise the search could wander to a flatter spot that's
        kinematically awkward or outside the leg's real reach). This is the actual
        ANYmal/Spot perceptive-locomotion mechanism -- search candidate footholds near
        the nominal one instead of only correcting the single nominal point's height
        (which is all compute_touchdown_world_for_traj_purpose_only/
        compute_swing_traj_and_touchdown did before this)."""
        offsets = np.arange(-FOOTHOLD_SEARCH_RADIUS, FOOTHOLD_SEARCH_RADIUS + 1e-9,
                             FOOTHOLD_GRID_STEP)
        dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
        dx, dy = dx.ravel(), dy.ravel()
        cx, cy = x_nom + dx, y_nom + dy

        eps = 0.02
        dzdx = (self._terrain_height_fn(cx + eps, cy) - self._terrain_height_fn(cx - eps, cy)) / (2 * eps)
        dzdy = (self._terrain_height_fn(cx, cy + eps) - self._terrain_height_fn(cx, cy - eps)) / (2 * eps)
        slope_mag = np.sqrt(dzdx**2 + dzdy**2)

        reg = FOOTHOLD_REG_WEIGHT * (dx**2 + dy**2) / FOOTHOLD_SEARCH_RADIUS**2
        cost = slope_mag + reg
        best = np.argmin(cost)
        return float(cx[best]), float(cy[best])

    def compute_touchdown_world_for_traj_purpose_only(self, go2, leg):
        pos = super().compute_touchdown_world_for_traj_purpose_only(go2, leg)
        pos = np.array(pos, dtype=float)
        if self._foothold_search:
            pos[0], pos[1] = self._best_foothold_xy(pos[0], pos[1])
        pos[2] = float(self._terrain_height_fn(pos[0], pos[1])) + 0.02
        return pos

    def compute_swing_traj_and_touchdown(self, go2, leg):
        _, pos_touchdown_world = super().compute_swing_traj_and_touchdown(go2, leg)
        pos_touchdown_world = np.array(pos_touchdown_world, dtype=float)
        if self._foothold_search:
            pos_touchdown_world[0], pos_touchdown_world[1] = self._best_foothold_xy(
                pos_touchdown_world[0], pos_touchdown_world[1])
        pos_touchdown_world[2] = float(self._terrain_height_fn(
            pos_touchdown_world[0], pos_touchdown_world[1])) + 0.02
        foot_pos, _ = go2.get_single_foot_state_in_world(leg)
        if self._swing_clearance_adaptive:
            h_sw = HEIGHT_SWING + 0.5 * abs(foot_pos[2] - pos_touchdown_world[2])
        else:
            h_sw = HEIGHT_SWING
        traj = self.make_swing_trajectory(foot_pos, pos_touchdown_world,
                                           self.swing_time, h_sw=h_sw)
        return traj, pos_touchdown_world


# Stage 11 MPC-only pivot: gait-style switching. gait.py's PHASE_OFFSET (leg order
# FL, FR, RL, RR -- see leg_controller.py's LEG_INDEX) is a module-level constant
# hardcoding a trot (diagonal pairs [FL,RR] and [FR,RL] share a phase). Real
# model-based quadruped controllers (MIT Cheetah 3, ANYmal) select a gait style by
# swapping this contact-schedule parametrization, not by learning -- no RL needed.
# Each entry below is (phase_offset, duty); duty is bundled per-style because
# "walk" genuinely needs a different duty (0.75, not GAIT_DUTY) to keep its
# single-leg-swing guarantee: with 4 legs' swing windows evenly spaced 0.25 apart
# in phase, a window wider than 0.25 (duty < 0.75) would make two legs swing at
# once, breaking the always-3-feet-down static-stability property that's the only
# reason to pick "walk" over a trot. "walk"'s phase_offset is chosen to reproduce
# this project's own Stage 0 scripted crawl's leg order (FL -> RR -> FR -> RL,
# see scripted_gait.py's LEGS/crawl_targets), not picked arbitrarily.
GAIT_HZ = 3
GAIT_DUTY = 0.6

GAIT_STYLE_PRESETS = {
    "trot":  (np.array([0.5, 0.0, 0.0, 0.5]), GAIT_DUTY),   # diagonal pairs (matches
                                                             # gait.py's own PHASE_OFFSET
                                                             # and current GAIT_DUTY
                                                             # exactly -- the default,
                                                             # bit-identical to every
                                                             # prior result).
    "pace":  (np.array([0.0, 0.5, 0.0, 0.5]), GAIT_DUTY),   # same-side pairs (FL&RL, FR&RR)
    "bound": (np.array([0.0, 0.0, 0.5, 0.5]), GAIT_DUTY),   # front/rear pairs (FL&FR, RL&RR)
    "walk":  (np.array([0.75, 0.25, 0.0, 0.5]), 0.75),      # sequential single-leg swing
}


class GaitStyleMixin:
    """Overrides Gait's compute_contact_table to read an instance-level phase offset
    (self._phase_offset, set by the subclass __init__ below) instead of gait.py's
    hardcoded module-level PHASE_OFFSET -- otherwise byte-identical to the parent
    formula. Mix in ahead of Gait (or a Gait subclass) so this override wins the MRO;
    compute_current_mask (unmodified, inherited) already calls self.compute_contact_table
    polymorphically, so overriding only this one method is sufficient for every caller
    (com_trajectory.py, leg_controller.py) to pick up the new schedule."""

    def compute_contact_table(self, t0: float, dt: float, N: int) -> np.ndarray:
        t = t0 + np.arange(N) * dt + dt / 2
        phases = np.mod(self._phase_offset[:, None] + t[None, :] / self.gait_period, 1.0)
        return (phases < self.gait_duty).astype(np.int32)


class TouchdownHeightMixin:
    """Experimental lever for bound: ties gait.py's hardcoded touchdown z=0.02 (see
    compute_touchdown_world_for_traj_purpose_only) to the same per-leg-pair phase used
    for the contact schedule, instead of a flat constant regardless of where the body
    is in its gait cycle. Independent of (and compatible with) GAIT_STYLE_PITCH_WEIGHT/
    GAIT_STYLE_PITCH_AMPLITUDE above -- those change the COST reference; this changes
    the PLANNED FOOT LEVER ARM fed into the predicted dynamics (Bd), which is the one
    place pitch actually could have mattered for Bd (see note below) but doesn't
    naturally, since compute_touchdown_world_for_traj_purpose_only only ever uses yaw
    (go2.R_z is built from yaw alone, never roll/pitch -- confirmed directly:
    go2_robot_data.py's R_z is cos/sin of `current_config.compute_euler_angle_world()[2]`
    i.e. yaw only) and a hardcoded z. This mixin changes z only; it does NOT attempt to
    route pitch into R_z (that would fight the controller's own yaw-only footstep-
    placement design more broadly, well beyond bound).

    compute_current_mask is called immediately before compute_touchdown_world_for_traj_
    purpose_only for every leg in ComTraj.generate_traj's per-step loop (the only
    caller), so caching the time there and reading it back here is reliable without
    needing a time argument on the touchdown method itself (its signature is fixed by
    the vendored Gait/TerrainAwareGait base classes)."""

    def compute_current_mask(self, time):
        self._last_touchdown_query_time = time
        return super().compute_current_mask(time)

    def compute_touchdown_world_for_traj_purpose_only(self, go2, leg):
        pos = super().compute_touchdown_world_for_traj_purpose_only(go2, leg)
        if self._touchdown_z_amplitude != 0.0:
            leg_phase = self._phase_offset[LEG_INDEX[leg]]
            phase = np.mod(leg_phase + self._last_touchdown_query_time / self.gait_period, 1.0)
            pos = pos.copy()
            pos[2] += self._touchdown_z_amplitude * np.sin(2.0 * np.pi * phase)
        return pos


class StyledGait(TouchdownHeightMixin, GaitStyleMixin, Gait):
    def __init__(self, frequency_hz, duty, phase_offset, touchdown_z_amplitude=0.0):
        self._phase_offset = np.asarray(phase_offset, dtype=float)
        self._touchdown_z_amplitude = touchdown_z_amplitude
        self._last_touchdown_query_time = 0.0
        super().__init__(frequency_hz, duty)


class StyledTerrainAwareGait(TouchdownHeightMixin, GaitStyleMixin, TerrainAwareGait):
    def __init__(self, frequency_hz, duty, phase_offset, terrain_height_fn, touchdown_z_amplitude=0.0,
                 foothold_search=False, swing_clearance_adaptive=False):
        self._phase_offset = np.asarray(phase_offset, dtype=float)
        self._touchdown_z_amplitude = touchdown_z_amplitude
        self._last_touchdown_query_time = 0.0
        super().__init__(frequency_hz, duty, terrain_height_fn, foothold_search, swing_clearance_adaptive)


class BoundAwareComTraj(ComTraj):
    """Gives the bound gait two genuine, phase-synced references instead of
    ComTraj.generate_traj's own hardcoded flat (zero pitch, constant height) reference
    used for every style:

    PITCH -- tied to the SAME phase used for the contact schedule (`phase_offset` --
    the front leg pair's own offset, since bound's front/rear pairs are exactly out of
    phase): nose dips one way while the front pair is in stance, the other way while
    the rear pair is. Follow-up to GAIT_STYLE_PITCH_WEIGHT's cost-softening fix below,
    which alone only got bound to 21% success -- softening the weight gives the QP room
    to deviate from zero pitch, but gives it no target to deviate TOWARD. Got bound to
    23% (see mpc/README.md).

    HEIGHT -- a genuine flight-phase vertical bounce, active only once GAIT_STYLE_PRESETS
    gives bound a real flight gap (duty < 0.5; see the duty sweep in mpc/README.md).
    Rather than guess a sinusoid's phase/sign the way the pitch reference above does,
    this derives the bump directly from the gait's own schedule: with front-pair offset
    0 and rear-pair offset 0.5 (bound's design), simple interval arithmetic gives exactly
    two "both pairs swinging" (flight) windows per period, each of width (0.5 - duty), at
    phase in [duty, 0.5) and [0.5+duty, 1.0). Within each window this places the same
    smooth bump-with-zero-endpoint-derivative shape leg_controller.py's own
    `make_swing_trajectory` already uses for swing-leg apex height (`64*s**3*(1-s)**3`),
    scaled by `height_amplitude` -- so the body is asked to rise during an actual flight
    window and nowhere else, not during either stance phase.

    Scope, by design, for BOTH references: only the QP's COST reference
    (rpy/pos/vel/omega_traj_world, what compute_x_ref_vec feeds the linear cost term) is
    patched, AFTER calling the unmodified parent generate_traj. Verified this doesn't
    matter for Ad/Bd (the predicted dynamics): `gait.py`'s own touchdown-prediction
    ignores body pitch (rotates the hip offset with a yaw-ONLY `R_z`) and body height
    (hardcodes `body_pos`'s z-component to 0, and the touchdown z itself to a flat
    constant) -- so neither reference ever reaches the foot-lever-arm/Bd computation
    regardless of patch ordering (confirmed directly; see mpc/README.md's
    "dynamics-model inconsistency" investigation). Patching after the parent call is
    therefore not a simplification, it's equivalent to patching before.
    """

    def __init__(self, go2, pitch_amplitude, pitch_phase_offset, gait_duty, height_amplitude=0.0):
        super().__init__(go2)
        self._pitch_amplitude = pitch_amplitude
        self._pitch_phase_offset = pitch_phase_offset
        self._gait_duty = gait_duty
        self._height_amplitude = height_amplitude

    def generate_traj(self, go2, gait, time_now, x_vel_des_body, y_vel_des_body,
                       z_pos_des_body, yaw_rate_des_body, time_step):
        super().generate_traj(go2, gait, time_now, x_vel_des_body, y_vel_des_body,
                               z_pos_des_body, yaw_rate_des_body, time_step)
        period = gait.gait_period
        t_abs = time_now + self.time
        two_pi = 2.0 * np.pi
        # Front-pair phase (bound's phase_offset[0] -- rear is always exactly 0.5 out
        # of phase with it, see GAIT_STYLE_PRESETS), shared by both references below.
        front_phase = np.mod(self._pitch_phase_offset + t_abs / period, 1.0)

        if self._pitch_amplitude != 0.0:
            self.rpy_traj_world[1, :] = self._pitch_amplitude * np.sin(two_pi * front_phase)
            self.omega_traj_world[1, :] = (self._pitch_amplitude * (two_pi / period)
                                            * np.cos(two_pi * front_phase))

        if self._height_amplitude != 0.0:
            assert self._gait_duty < 0.5, (
                "height bounce needs a genuine flight gap (duty < 0.5); "
                f"got duty={self._gait_duty}")
            d = self._gait_duty
            w = 0.5 - d  # flight-window width, in units of phase (fraction of period)

            def bump(s):
                return 64.0 * s**3 * (1.0 - s)**3

            def dbump_ds(s):
                return 192.0 * s**2 * (1.0 - s)**2 * (1.0 - 2.0 * s)

            s1 = np.clip((front_phase - d) / w, 0.0, 1.0)
            in1 = (front_phase >= d) & (front_phase < 0.5)
            s2 = np.clip((front_phase - (0.5 + d)) / w, 0.0, 1.0)
            in2 = front_phase >= (0.5 + d)

            b = np.where(in1, bump(s1), np.where(in2, bump(s2), 0.0))
            db_ds = np.where(in1, dbump_ds(s1), np.where(in2, dbump_ds(s2), 0.0))
            db_dt = db_ds / (w * period)

            self.pos_traj_world[2, :] += self._height_amplitude * b
            self.vel_traj_world[2, :] += self._height_amplitude * db_dt


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

class TerrainAwareCentroidalMPC(CentroidalMPC):
    """Patches the submodule's own CentroidalMPC (subclassed, not edited in place -- same
    discipline as TerrainAwareGait above) so the QP's friction-cone constraints are
    oriented around the REAL local terrain surface normal instead of always assuming the
    ground normal is world-up.

    Root cause this fixes: `CentroidalMPC._precompute_friction_matrix` builds the
    friction pyramid purely from world-frame force components (`|fx| <= mu*fz`,
    `|fy| <= mu*fz`), i.e. it always treats world Z as the contact normal direction and
    world X/Y as the tangent plane. That's only correct on flat ground. On a slope, the
    real friction cone is tilted with the terrain: commanding a world-frame force that
    satisfies the (wrong) flat-ground cone can still violate the REAL cone (risking an
    actual slip when the commanded torque is realized in MuJoCo's own contact physics,
    not just a theoretical QP-feasibility issue) -- or can needlessly exclude force
    combinations the real terrain would have supported, falsely narrowing the QP's own
    feasible region (a likely contributor to TerrainAwareGait's still-41% qp_failed_rate,
    see mpc/README.md). This is exactly the kind of terrain-geometry fact real quadruped
    controllers fold into their contact model, not just footstep placement.

    `_precompute_friction_matrix` is overridden to build each face from a signed tangent
    direction and the local normal (`sign*(t . F) - MU*(n . F) <= 0`) instead of hardcoded
    world axes -- on flat ground (n=[0,0,1], t1=[1,0,0], t2=[0,1,0]) this reduces to
    EXACTLY the original formula (verified below). Every row explicitly stores all three
    (fx, fy, fz) columns even when a coefficient is exactly 0.0 (e.g. the flat-ground
    case), so the constraint matrix's SPARSITY stays fixed across different normals --
    required because the OSQP solver object is only ever constructed once per episode
    (`_build_sparse_matrix`, called from `CentroidalMPC.__init__`) and expects the same
    nonzero pattern on every subsequent solve.

    The terrain normal is recomputed every `solve_QP` call (not just once per episode)
    from the robot's CURRENT position via a finite-difference gradient of the actual
    terrain heightfield (`terrain_normal_fn`, see `MPCResidualEnv._terrain_normal`) --
    needed because the terrain's local slope genuinely changes as the robot walks from
    the flat start pad onto a ramp/stairs, not just once at episode start.

    Known remaining approximation, not fixed here: the `fz_min` stance-force lower bound
    in `_compute_bounds` still bounds the raw WORLD-z force component, not the force
    along the (now correctly oriented) local normal -- left as a box constraint since
    reformulating it as a general inequality would need the same per-row sparsity
    surgery as the friction pyramid, for a much smaller physical effect at the slope
    angles tested here (world-z and true-normal force differ by a factor of cos(slope),
    e.g. 2% at 12deg) than getting the dominant friction-cone orientation right.
    """

    def __init__(self, go2, traj, terrain_normal_fn):
        self._terrain_normal_fn = terrain_normal_fn
        self._current_normal = np.array([0.0, 0.0, 1.0])  # flat default for __init__'s own build
        super().__init__(go2, traj)

    def solve_QP(self, go2, traj, verbose: bool = False):
        pos = go2.current_config.base_pos
        n = np.asarray(self._terrain_normal_fn(float(pos[0]), float(pos[1])), dtype=float)
        self._current_normal = n / np.linalg.norm(n)
        return super().solve_QP(go2, traj, verbose)

    def _precompute_friction_matrix(self, traj):
        n = self._current_normal
        world_x = np.array([1.0, 0.0, 0.0])
        ref = np.array([0.0, 1.0, 0.0]) if abs(np.dot(n, world_x)) > 0.9 else world_x
        t1 = ref - np.dot(ref, n) * n
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        rows, cols, vals = [], [], []
        baseU = self.N * NX
        r0 = 0
        for k in range(self.N):
            uk0 = baseU + k * NU
            for leg in range(4):
                fx, fy, fz = 3 * leg, 3 * leg + 1, 3 * leg + 2
                for sign, t in ((1.0, t1), (-1.0, t1), (1.0, t2), (-1.0, t2)):
                    coeffs = sign * t - MU * n
                    rows.extend([r0, r0, r0])
                    cols.extend([uk0 + fx, uk0 + fy, uk0 + fz])
                    vals.extend([float(coeffs[0]), float(coeffs[1]), float(coeffs[2])])
                    r0 += 1

        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(r0, self.nvars))
        return self._scipy_to_casadi(A_sp)

    def _assemble_A_matrix(self, Ad, Bd):
        # Base class reads the STALE self.A_ineq_static (computed once). Recompute fresh
        # every call instead, using whatever self._current_normal solve_QP just set.
        big_minus_Ad, big_minus_Bd = self.dyn_builder(Ad, Bd)
        term_Ad = self.S_block @ big_minus_Ad
        A_eq = ca.horzcat(self.I_block + term_Ad, big_minus_Bd)
        A_ineq = self._precompute_friction_matrix(None)
        return ca.vertcat(A_eq, A_ineq)


# Bound-gait pitch fix, two complementary cost-side levers. ComTraj.generate_traj
# hardcodes a FLAT (zero) roll/pitch reference over the whole MPC horizon regardless of
# gait style (see com_trajectory.py: rpy_traj_world[0,:] = rpy_traj_world[1,:] = 0.0).
# Trot/pace never conflict with this -- both always keep some left-right- or
# diagonally-balanced support, so there's no systematic pitching moment to fight. Bound
# alternates "both front feet down, both rear up" and back, which creates a real
# pitching moment every half-cycle; demanding the QP hold a rigid zero-pitch reference
# against that moment using only 2 legs' worth of stance force is why bound measured
# 94% qp_failed (see mpc/README.md).
#
# Lever 1 (below): soften the state-cost weight on pitch tracking (COST_MATRIX_Q's
# index 4) so the QP has room to let the body pitch along with the physics rather than
# fighting it. Swept empirically (20/5/2/1/0.5/0.1/0 -> success 0.06/0.15/0.17/0.21/
# 0.17/0.17/0.12): a real, unimodal peak at weight=1.0, not "softer is always better"
# (too little pitch-tracking pull lets pitch drift uncontrolled instead). On its own
# this got bound from 6% to 21% success -- real, but far from trot's 94%, since
# softening gives the QP room to deviate from zero pitch but no target to deviate
# TOWARD, leaving the deviation underdetermined.
#
# Lever 2 (BoundAwareComTraj above): gives the QP that missing target -- a genuine,
# phase-synced sinusoidal pitch reference, swept together with lever 1 in
# GAIT_STYLE_PITCH_AMPLITUDE below. Both levers left at None/0.0 (original
# COST_MATRIX_Q / no reference) for every style except bound, keeping every non-bound
# result exactly reproducible.
GAIT_STYLE_PITCH_WEIGHT = {
    "trot": None,
    "pace": None,
    "bound": 1.0,   # empirically the best of 20/5/2/1/0.5/0.1/0 swept against
                     # COST_MATRIX_Q's original 20 (see sweep above).
    "walk": None,
}

GAIT_STYLE_PITCH_AMPLITUDE = {
    "trot": 0.0,
    "pace": 0.0,
    "bound": 0.3,   # best of 0/+-0.05/+-0.15/+-0.3 rad swept against weight=1.0 (above):
                     # success 0.21/0.15/0.02/0.19/0.00/0.23/0.00 -- sign matters a lot
                     # (negative is unambiguously wrong, gets catastrophic as |amplitude|
                     # grows), positive is safe and helps episode length (783->1069
                     # steps) but only marginally improves full-episode success rate
                     # (0.21->0.23, not clearly beyond noise at n=48). See mpc/README.md.
    "walk": 0.0,
}

# TouchdownHeightMixin lever: ties gait.py's hardcoded touchdown z=0.02 to each leg
# pair's own phase instead of a flat constant. 0.0 for every style until swept for
# bound (see mpc/README.md); kept as a separate dict since it's an independent,
# orthogonal lever from the two above (changes the predicted foot lever arm fed into
# Bd, not the QP's cost reference).
GAIT_STYLE_TOUCHDOWN_Z_AMPLITUDE = {
    "trot": 0.0,
    "pace": 0.0,
    "bound": 0.0,
    "walk": 0.0,
}

# BoundAwareComTraj's flight-phase vertical-bounce lever (see its docstring). Only
# meaningful once GAIT_STYLE_PRESETS["bound"]'s duty is < 0.5 (a genuine flight gap
# exists); 0.0 for every style pending the sweep. See mpc/README.md.
GAIT_STYLE_HEIGHT_AMPLITUDE = {
    "trot": 0.0,
    "pace": 0.0,
    "bound": 0.0,
    "walk": 0.0,
}


class CostOverrideMixin:
    """Rebuilds the QP's Hessian/solver with a different state-cost weight matrix than
    the submodule's own COST_MATRIX_Q. Mix in ahead of CentroidalMPC (or a CentroidalMPC
    subclass) in the MRO. The base __init__ already builds H/the solver once from
    COST_MATRIX_Q (no constructor seam to supply a different Q before that happens, and
    the vendored submodule isn't edited in place) -- so this calls the unmodified parent
    __init__ first, then overwrites self.Q and calls the still-unmodified
    _build_sparse_matrix again to rebuild H/the solver from the new Q. Safe to call
    twice: _build_sparse_matrix recomputes everything from current self.Q/self.N/
    self._assemble_A_matrix with no stale-state accumulation, and Q is fixed for the
    rest of the episode (never changed per-solve), so the final sparsity this produces
    is the one every subsequent solve_QP call will actually match."""

    def __init__(self, go2, traj, cost_q, *args, **kwargs):
        super().__init__(go2, traj, *args, **kwargs)
        self.Q = np.asarray(cost_q, dtype=float)
        self._build_sparse_matrix(traj, verbose=False)


class CostOverrideCentroidalMPC(CostOverrideMixin, CentroidalMPC):
    pass


class CostOverrideTerrainAwareCentroidalMPC(CostOverrideMixin, TerrainAwareCentroidalMPC):
    pass


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
        terrain_footstep_planning: bool = False,  # make MPC's OWN footstep planner (Gait's
                                               # touchdown-height computation) terrain-aware
                                               # instead of hardcoding flat ground -- see
                                               # TerrainAwareGait above. Independent of
                                               # use_terrain_heightmap (that's the RL residual's
                                               # observation; this is the base controller's own
                                               # planning). Default False reproduces terrain_v1-
                                               # v4's exact prior (flat-footstep-plan) behavior.
        terrain_friction_cone: bool = False,  # orient the MPC's QP friction-cone constraints
                                               # around the REAL local terrain normal instead of
                                               # always assuming world-up -- see
                                               # TerrainAwareCentroidalMPC above. Independent of
                                               # terrain_footstep_planning (that fixes WHERE feet
                                               # land; this fixes what forces are assumed
                                               # achievable once planted). Default False
                                               # reproduces every prior variant's exact behavior.
        terrain_foothold_optimization: bool = False,  # search nearby (x, y) candidate footholds
                                               # for a flatter spot instead of only correcting the
                                               # nominal Raibert point's height -- see
                                               # TerrainAwareGait._best_foothold_xy. Requires
                                               # terrain_footstep_planning (needs the same terrain-
                                               # height function); a no-op otherwise. Default False
                                               # reproduces every prior variant's exact behavior.
        terrain_swing_clearance: bool = False,  # make the swing-leg apex height adapt to the
                                               # takeoff/touchdown height difference instead of a
                                               # fixed HEIGHT_SWING -- see TerrainAwareGait.
                                               # compute_swing_traj_and_touchdown. Requires
                                               # terrain_footstep_planning; a no-op otherwise.
                                               # Default False reproduces every prior variant's
                                               # exact behavior.
        gait_style: str = "trot",             # one of GAIT_STYLE_PRESETS -- selects the
                                               # contact-schedule parametrization (trot/pace/
                                               # bound/walk), a classical, non-learned choice.
                                               # Default "trot" uses the exact original
                                               # phase_offset/duty, reproducing every prior
                                               # result exactly (verified bit-for-bit).
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
        if gait_style not in GAIT_STYLE_PRESETS:
            raise ValueError(f"gait_style must be one of {list(GAIT_STYLE_PRESETS)}, got {gait_style!r}")
        self.gait_style = gait_style
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
        # self._terrain_height is a bound method referencing `self` -- it's looked up
        # dynamically each call, so passing it here (before self._terrain_h etc. are set
        # up below) is safe: by the time TerrainAwareGait actually calls it (from
        # reset()/step(), always after _generate_terrain has populated self._terrain_h),
        # it reflects the current episode's terrain. terrain_footstep_planning gates this
        # independently of terrain_enabled so the earlier terrain_v1-v4 behavior (terrain
        # present, but MPC's own footstep planner still flat-ground-blind) stays
        # reproducible without this fix.
        self.terrain_footstep_planning = terrain_footstep_planning and self.terrain_enabled
        self.terrain_foothold_optimization = terrain_foothold_optimization and self.terrain_footstep_planning
        self.terrain_swing_clearance = terrain_swing_clearance and self.terrain_footstep_planning
        phase_offset, gait_duty = GAIT_STYLE_PRESETS[self.gait_style]
        touchdown_z_amplitude = GAIT_STYLE_TOUCHDOWN_Z_AMPLITUDE[self.gait_style]
        self.gait = (StyledTerrainAwareGait(GAIT_HZ, gait_duty, phase_offset, self._terrain_height,
                                             touchdown_z_amplitude, self.terrain_foothold_optimization,
                                             self.terrain_swing_clearance)
                     if self.terrain_footstep_planning else
                     StyledGait(GAIT_HZ, gait_duty, phase_offset, touchdown_z_amplitude))
        # Same late-binding safety as self._terrain_height above -- self._terrain_normal is
        # a bound method, only ever called from solve_QP() well after _generate_terrain has run.
        self.terrain_friction_cone = terrain_friction_cone and self.terrain_enabled
        pitch_amplitude = GAIT_STYLE_PITCH_AMPLITUDE[self.gait_style]
        height_amplitude = GAIT_STYLE_HEIGHT_AMPLITUDE[self.gait_style]
        self.traj = (BoundAwareComTraj(self.go2, pitch_amplitude, float(phase_offset[0]),
                                        gait_duty, height_amplitude)
                     if (pitch_amplitude != 0.0 or height_amplitude != 0.0) else ComTraj(self.go2))
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
        pitch_weight = GAIT_STYLE_PITCH_WEIGHT[self.gait_style]
        if pitch_weight is not None:
            # COST_MATRIX_Q is int-dtype (np.diag of an all-int list) -- .copy() alone
            # preserves that dtype, silently truncating a non-integer pitch_weight to 0
            # on assignment (confirmed directly: 0.5 and 0.1 both became exactly 0.0,
            # producing bit-identical, wrong results). astype(float) first avoids this.
            cost_q = COST_MATRIX_Q.astype(float)
            cost_q[4, 4] = pitch_weight
            if self.terrain_friction_cone:
                self.mpc = CostOverrideTerrainAwareCentroidalMPC(
                    self.go2, self.traj, cost_q, self._terrain_normal)
            else:
                self.mpc = CostOverrideCentroidalMPC(self.go2, self.traj, cost_q)
        elif self.terrain_friction_cone:
            self.mpc = TerrainAwareCentroidalMPC(self.go2, self.traj, self._terrain_normal)
        else:
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

    def _terrain_normal(self, x: float, y: float) -> np.ndarray:
        """Local terrain surface unit normal (world frame) at (x, y), via a central-difference
        gradient of _terrain_height. [0, 0, 1] on flat ground / terrain disabled, matching
        _terrain_height's own flat-ground return. Feeds TerrainAwareCentroidalMPC's friction-cone
        orientation -- recomputed every MPC solve (not cached), since the real local slope
        changes as the robot walks from the flat start pad onto a ramp/stairs."""
        eps = 0.03
        dzdx = (self._terrain_height(x + eps, y) - self._terrain_height(x - eps, y)) / (2 * eps)
        dzdy = (self._terrain_height(x, y + eps) - self._terrain_height(x, y - eps)) / (2 * eps)
        n = np.array([-dzdx, -dzdy, 1.0])
        return n / np.linalg.norm(n)

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

        # base_pos/yaw: same keys/shape envs/go1_env.py's own info dict already exposes, added so
        # navigate.py's existing compute_nav_command() outer loop (world position + heading each
        # step) works against this env unmodified -- yaw via the identical formula _local_heightmap
        # already uses elsewhere in this file (MuJoCo's own qpos quat order is w,x,y,z here).
        qw, qx, qy, qz = self.mujoco_go2.data.qpos[3:7]
        yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
        info = {
            "base_height": base_height, "roll": roll, "pitch": pitch, "qp_failed": qp_failed,
            "v_body": v_body.copy(), "w_body": w_body.copy(),
            "base_pos": self.mujoco_go2.data.qpos[0:2].copy(), "yaw": yaw,
            # Same key ("reward_components") and flat-dict-of-floats shape train.py's own
            # RewardComponentLoggingCallback already expects, so it's reusable unmodified here.
            "reward_components": {
                "speed": r_speed, "lateral": r_lateral, "yaw_rate": r_yaw_rate, "effort": r_effort,
            },
        }
        return obs, float(reward), terminated, truncated, info

    def set_command(self, target_speed: float, target_lateral: float = 0.0) -> None:
        """Overwrite the commanded forward/lateral velocity directly, bypassing reset()'s own
        per-episode randomization -- mirrors envs/go1_env.py's own set_command exactly (same name/
        signature), so navigate.py's outer loop can drive either env interchangeably."""
        self.target_speed = float(target_speed)
        self.target_lateral = float(target_lateral)
