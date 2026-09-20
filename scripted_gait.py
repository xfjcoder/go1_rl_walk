"""
Hand-designed statically-stable "crawl" gait for the Go1 -- NO RL involved.

Purpose: a physics/model sanity check and a baseline. If this walks, the
MJCF model, PD gains and torque limits can support a real forward gait, so
any later RL failure is a reward/training problem, not a model problem.

How it works
  * One leg swings at a time (order FL -> RR -> FR -> RL), the other three
    sweep backward at body speed `v` (that sweep is what pushes the body).
  * Before each lift the whole body is shifted over the centroid of the three
    remaining stance feet. Without this the trunk CoM sits outside the
    support triangle for rear-leg lifts (measured: 3-4 cm outside), the body
    tips, stance feet slip and the robot drifts backward instead of forward.
  * Foot targets are given in the body frame and converted to (hip, thigh,
    calf) by a closed-form 3D IK, verified against MuJoCo forward kinematics.

Robustness (24 randomized trials: friction 0.7-1.2x, trunk mass +-10%, joint
noise 0.03 rad, 30 s): the defaults below, with kp=150 / kd=3, gave 0 falls,
~0.096 m/s forward, ~1.4 deg yaw drift.
"""
import numpy as np

# Geometry copied from assets/go1.xml (keep in sync).
L1 = L2 = 0.213                                   # thigh / calf length
LEGS = ["FR", "FL", "RR", "RL"]                   # joint order used everywhere: hip, thigh, calf per leg
HIP_XY = {"FR": (0.1881, -0.04675), "FL": (0.1881, 0.04675),
          "RR": (-0.1881, -0.04675), "RL": (-0.1881, 0.04675)}   # hip joint origin in trunk frame
HIP_OFF = {"FR": -0.08, "FL": 0.08, "RR": -0.08, "RL": 0.08}      # thigh y-offset from hip joint
DEFAULT_JOINT_POS = np.array([0.0, 0.6, -1.2] * 4)                # standing pose
STAND_DEPTH = 0.3516                                               # hip-to-foot height when standing
COM_B = (-0.022, 0.0)                                              # whole-robot CoM in trunk frame (xy)

SWING_ORDER = ["FL", "RR", "FR", "RL"]


def leg_ik(leg, x, y, z):
    """Foot (x fwd, y left, z up) relative to the HIP JOINT origin -> (hip, thigh, calf)."""
    off = HIP_OFF[leg]
    zl = -np.sqrt(max(y * y + z * z - off * off, 1e-9))          # vertical reach inside the thigh/calf plane
    hip = np.arctan2(z, y) - np.arctan2(zl, off)
    hip = (hip + np.pi) % (2 * np.pi) - np.pi
    u, w = -x, -zl
    d = np.hypot(u, w)
    cos2 = np.clip((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    calf = -np.arccos(cos2)
    thigh = np.arctan2(u, w) - np.arctan2(L2 * np.sin(calf), L1 + L2 * np.cos(calf))
    return hip, thigh, calf


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def crawl_targets(t, T=3.0, v=0.12, swing_frac=0.3, ramp_frac=0.4, swing_height=0.06,
                  t_start=3.0, t_blend=2.0, max_shift=0.08):
    """
    12 joint position targets (FR, FL, RR, RL x hip, thigh, calf) at sim time t.

    T            gait cycle (s); each leg gets a slot of T/4
    v            commanded body speed (m/s) -- stance feet sweep back at this rate
    swing_frac   fraction of a slot a leg spends in the air
    ramp_frac    fraction of a slot used to shift the body before a lift / back after it
    swing_height peak foot lift (m)
    t_start      time the first swing begins; before that all four feet stay down
    t_blend      time to blend from the standing pose into the gait's stagger pattern
    """
    slot = T / 4
    ts = swing_frac * slot                        # swing duration
    tr = ramp_frac * slot                         # body-shift ramp duration
    tst = T - ts                                  # stance duration
    tt = t - t_start
    nom = {l: np.array([HIP_XY[l][0], HIP_XY[l][1] + HIP_OFF[l]]) for l in LEGS}   # standing foot xy, trunk frame

    dx, dz = {}, {}
    for i, leg in enumerate(SWING_ORDER):
        ph = (max(tt, 0.0) - i * slot) % T        # time since this leg's swing began (frozen before start)
        if ph < ts:                               # swing: foot arcs forward relative to the body
            f = ph / ts
            dx[leg] = -v * tst / 2 + f * v * tst
            dz[leg] = swing_height * np.sin(np.pi * f)
        else:                                     # stance: foot sweeps backward at body speed
            dx[leg] = v * tst / 2 - (ph - ts) * v
            dz[leg] = 0.0

    shift = np.zeros(2)                           # body displacement relative to the feet
    for i, leg in enumerate(SWING_ORDER):
        if tt < -tr:
            continue
        ph = (tt - i * slot + tr) % T - tr        # in [-tr, T - tr)
        if ph >= ts + tr:
            continue
        w = _smoothstep((ph + tr) / tr) * (1 - _smoothstep((ph - ts) / tr))
        stance = [m for m in LEGS if m != leg]
        centroid = np.mean([nom[m] + np.array([dx[m], 0.0]) for m in stance], axis=0)
        s = centroid - np.array(COM_B)            # move the CoM onto the stance centroid
        n = np.linalg.norm(s)
        if n > max_shift:
            s *= max_shift / n
        shift += w * s

    targets = np.zeros(12)
    for k, leg in enumerate(LEGS):
        fx = nom[leg][0] + dx[leg] - shift[0]     # feet move opposite to the body shift
        fy = nom[leg][1] - shift[1]
        fz = -STAND_DEPTH + dz[leg]
        targets[3 * k:3 * k + 3] = leg_ik(leg, fx - HIP_XY[leg][0], fy - HIP_XY[leg][1], fz)

    b = _smoothstep(t / t_blend) if t_blend > 0 else 1.0
    if b < 1.0:
        targets = (1 - b) * DEFAULT_JOINT_POS + b * targets
    return targets
