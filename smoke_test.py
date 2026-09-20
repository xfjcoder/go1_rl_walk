"""
Quick sanity check: loads go1.xml, holds the standing keyframe with a
simple PD controller, and steps the sim. Only needs `mujoco` installed
(no gymnasium/stable-baselines3/torch required). Run this first to
confirm the model compiles and stands up before starting RL training.

Usage:
    python smoke_test.py                       # headless, prints trunk height over time
    python smoke_test.py --view                 # opens interactive MuJoCo viewer (needs GLFW)
    python smoke_test.py --record stand.gif      # offscreen render -> GIF, no window/GLFW needed
    python smoke_test.py --record walk.gif --walk --seconds 30   # scripted crawl gait (see scripted_gait.py)
    python smoke_test.py --record walk.gif --walk --walk-style trot   # old scripted IK trot, NO RL involved at all --
                                                       # tests whether the robot/physics can walk
                                                       # via a hand-designed gait before trusting
                                                       # RL to discover one from scratch.

Note on --view: MuJoCo's interactive viewer uses GLFW, whose Wayland
backend segfaults on some Linux setups. If `--view` crashes, try:
    WAYLAND_DISPLAY= python3 smoke_test.py --view
or use `--record` instead, which never opens a window.
"""
import argparse
import numpy as np
import mujoco

from scripted_gait import crawl_targets

DEFAULT_JOINT_POS = np.array([0.0, 0.6, -1.2] * 4)

# Leg segment lengths (thigh, calf), matching assets/go1.xml
L1 = L2 = 0.213

GAIT_PHASE_OFFSETS = {
    # "trot": diagonal pairs together (FR+RL, FL+RR), duty_factor=0.5 (2 feet down at a
    # time) -- only DYNAMICALLY stable, confirmed to require active balance correction
    # that a simple hip PD controller couldn't provide (repeatedly flipped over).
    "trot": {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0},
    # "static": one leg swings at a time in sequence (FL -> RL -> FR -> RR), duty_factor=
    # 0.75 (3 feet always down) -- STATICALLY stable by construction: the center of mass
    # stays within the support triangle at all times, no active balance needed. Standard
    # first gait for getting any legged robot walking before attempting a dynamic trot.
    "static": {"RR": 0.0, "FR": 0.25, "RL": 0.5, "FL": 0.75},
}
DUTY_FACTOR = {"trot": 0.5, "static": 0.75}


def inverse_kinematics(x_target: float, z_target: float):
    """
    Given a desired foot position (x, z) relative to the hip (x=forward,
    z=down as negative), return (thigh_angle, calf_angle) in this project's
    joint convention. Verified against the known-correct standing pose
    (thigh=0.6, calf=-1.2 -> foot at x=0, z=-0.3516): reproduces it exactly.
    """
    u, w = -x_target, -z_target
    d = np.sqrt(u ** 2 + w ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = -np.arccos(cos_theta2)
    theta1 = np.arctan2(u, w) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return theta1, theta2


def trot_foot_position(phase: float, stride_length: float, swing_height: float,
                        stand_depth: float, duty_factor: float = 0.5):
    """
    Foot (x, z) relative to hip, for a given point in this leg's own gait
    cycle (phase in [0, 1)). duty_factor is the fraction of the cycle spent
    in stance (0.5 for a trot with 2 legs down at a time; 0.75 for a static
    crawl with 3 legs down at a time).

    Stance (phase < duty_factor): foot on the ground, sweeping backward (x:
    +half stride -> -half stride) -- this is what actually pushes the body
    forward, since the hip/thigh rotates while the foot is anchored.

    Swing (phase >= duty_factor): foot lifts off, arcs forward back to the
    start of the next stance (x: -half stride -> +half stride), height
    rising and falling in a half-sine arc peaking at swing_height.
    """
    half_stride = stride_length / 2.0
    if phase < duty_factor:
        frac = phase / duty_factor
        x = half_stride - frac * stride_length
        z = -stand_depth
    else:
        frac = (phase - duty_factor) / (1 - duty_factor)
        x = -half_stride + frac * stride_length
        z = -stand_depth + swing_height * np.sin(np.pi * frac)
    return x, z


def trot_joint_targets(t: float, gait_period: float, stride_length: float,
                        swing_height: float, stand_depth: float, gait_style: str = "trot"):
    """Full 12-dim joint target array (hip=0, thigh/calf via IK) for time t."""
    targets = np.zeros(12)
    leg_order = ["FR", "FL", "RR", "RL"]
    offsets = GAIT_PHASE_OFFSETS[gait_style]
    duty_factor = DUTY_FACTOR[gait_style]
    for i, leg in enumerate(leg_order):
        phase = ((t / gait_period) + offsets[leg]) % 1.0
        x, z = trot_foot_position(phase, stride_length, swing_height, stand_depth, duty_factor)
        thigh, calf = inverse_kinematics(x, z)
        targets[3 * i + 0] = 0.0     # hip
        targets[3 * i + 1] = thigh
        targets[3 * i + 2] = calf
    return targets


def print_joint_report(data, qpos_adr, joint_names, target_pos, model=None):
    """Print the ACTUAL settled joint angles (not the commanded target), so
    front/rear differences can be checked precisely instead of relying on
    visual impression from a perspective camera, which can make legs on the
    far side of frame look subtly different even when they're mechanically
    identical."""
    actual = data.qpos[qpos_adr]
    print(f"\nActual settled joint angles (rad) vs commanded target:")
    for name, val, target in zip(joint_names, actual, target_pos):
        print(f"  {name:16s} actual={val:+.4f}  target={target:+.4f}  error={val - target:+.4f}")
    front_thigh_err = np.mean([actual[1] - target_pos[1], actual[4] - target_pos[4]])
    rear_thigh_err = np.mean([actual[7] - target_pos[7], actual[10] - target_pos[10]])
    print(f"\nMean front thigh tracking error: {front_thigh_err:+.4f} rad")
    print(f"Mean rear  thigh tracking error: {rear_thigh_err:+.4f} rad")
    print(f"Front vs rear difference: {abs(front_thigh_err - rear_thigh_err):.4f} rad "
          f"({np.degrees(abs(front_thigh_err - rear_thigh_err)):.2f} degrees)")
    right_hip_actual = [actual[0], actual[6]]
    left_hip_actual = [actual[3], actual[9]]
    print(f"\nRight hip (FR, RR) actual: {right_hip_actual[0]:+.4f}, {right_hip_actual[1]:+.4f}")
    print(f"Left  hip (FL, RL) actual: {left_hip_actual[0]:+.4f}, {left_hip_actual[1]:+.4f}")

    if model is not None:
        # The real test: did the FEET actually move outward (away from center,
        # more negative Y for right legs, more positive Y for left legs) as a
        # symmetric splay would require? Joint angles alone can't answer this --
        # same-sign joint values could mean either correct mirroring OR the same
        # bug thigh had (both sides moving the same absolute world direction).
        print("\nFoot world Y-position (body-frame width axis; more negative = further "
              "right, more positive = further left):")
        for site_name in ["FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"]:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            y = data.site_xpos[site_id][1]
            print(f"  {site_name:16s} y={y:+.4f} m")
        print("\nFor a symmetric OUTWARD splay: FR/RR (right) should move MORE NEGATIVE, "
              "FL/RL (left) should move MORE POSITIVE, relative to their standing values. "
              "If instead right and left both shifted the SAME direction, that's the mirroring bug.")


def get_roll(quat):
    """Roll angle (rotation about the forward/X axis) from a (w,x,y,z) quaternion."""
    w, x, y, z = quat
    return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="assets/go1.xml")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--record", type=str, default=None,
                         help="Path to save an offscreen-rendered GIF (bypasses GLFW entirely)")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--test-hip", type=float, default=None,
                         help="Diagnostic: command ALL FOUR hip joints to this same offset from "
                              "default (e.g. 0.2) and hold thigh/calf at their normal defaults. "
                              "Hip bodies for left/right legs differ only in Y-position with no "
                              "rotation applied (the same setup that caused the front/rear thigh "
                              "mirroring bug), so 'positive' hip abduction may mean 'outward' for "
                              "one side and 'inward' (swinging under the body) for the other. "
                              "Watch the recorded GIF: do all four legs splay the SAME relative "
                              "direction (all outward or all inward), or do left and right move "
                              "oppositely? This tells us whether the same class of bug exists here.")
    parser.add_argument("--walk", action="store_true",
                         help="Drive the robot with a hand-designed IK gait trajectory instead of "
                              "holding a static pose -- NO RL involved. Tests whether the physics "
                              "(torque limits, masses, PD gains) can support a real walking gait at "
                              "all, decoupling that question from 'can RL learn one.'")
    parser.add_argument("--walk-style", type=str, default="crawl", choices=["crawl", "trot", "static"],
                         help="'crawl' (default): one leg at a time WITH a body weight shift over the "
                              "support triangle (scripted_gait.py) -- the only style here that actually "
                              "walks forward. 'static': one leg swings at a time, 3 always down -- STATICALLY "
                              "stable, no active balance needed. Try this first. 'trot': diagonal "
                              "pairs (2 down at a time) -- only dynamically stable, confirmed to "
                              "flip over with a simple hip PD balance controller.")
    parser.add_argument("--crawl-speed", type=float, default=0.12,
                         help="Crawl style: commanded body speed (m/s). Measured speed is ~80%% of this.")
    parser.add_argument("--crawl-period", type=float, default=3.0,
                         help="Crawl style: seconds per full 4-leg cycle. The first swing starts at t=3 s, "
                              "so use --seconds 20 or more.")
    parser.add_argument("--crawl-swing-height", type=float, default=0.06, help="Crawl style: foot lift (m).")
    parser.add_argument("--walk-gait-period", type=float, default=1.2,
                         help="Seconds per stride cycle. Static gaits are typically slower than "
                              "trots since only one leg moves at a time.")
    parser.add_argument("--stride-length", type=float, default=0.10, help="Meters, fore-aft foot travel.")
    parser.add_argument("--swing-height", type=float, default=0.04, help="Meters, foot lift during swing.")
    parser.add_argument("--stand-depth", type=float, default=0.3516,
                         help="Meters, hip-to-foot vertical distance at rest (matches the default "
                              "standing pose: thigh=0.6, calf=-1.2).")
    parser.add_argument("--roll-gain", type=float, default=0.0,
                         help="Active lateral balance correction: adjusts hip abduction "
                              "differentially (right side vs left side) proportional to measured "
                              "body roll, to counteract the side-to-side rocking a diagonal-support "
                              "trot naturally induces. The scripted trot previously held hip "
                              "rigidly at 0 with no such correction and rocked/toppled regardless "
                              "of stride length -- this was the actual missing piece, not torque. "
                              "Set to 0 to disable and reproduce the old (rocking) behavior. If "
                              "the correction makes rocking WORSE (runs away to a full flip "
                              "instead of damping out), the sign is backwards -- try a negative "
                              "value.")
    parser.add_argument("--max-roll-correction", type=float, default=0.3,
                         help="Safety clamp (rad) on the roll-correction magnitude. Without this, "
                              "a wrong-sign or too-large gain can create a runaway positive "
                              "feedback loop (small roll -> destabilizing correction -> bigger "
                              "roll -> ...) that flips the robot over -- confirmed: an unclamped "
                              "correction reached 2*pi radians once roll approached pi (i.e. "
                              "already upside down) before this clamp was added.")
    parser.add_argument("--roll-rate-gain", type=float, default=0.5,
                         help="Damping term (D) on the roll correction, using measured roll "
                              "angular velocity. Pure proportional (P-only) feedback on roll angle "
                              "alone tends to overshoot and oscillate rather than settle -- "
                              "confirmed: flipping still occurred with EITHER sign of --roll-gain "
                              "alone, just in a different failure pattern. Adding rate damping "
                              "(making this a proper PD roll controller, matching how the joint "
                              "controllers already work) should let it settle instead of "
                              "oscillating into a flip.")
    parser.add_argument("--kp", type=float, default=None,
                         help="PD position gain (N*m/rad). Default 80 (150 for --walk-style crawl). "
                              "Was 40.0 originally; direct A/B testing showed 40.0 causes ~20 degree "
                              "tracking error under load (rear legs specifically), 80.0 drops that to "
                              "~1-2 degrees.")
    parser.add_argument("--kd", type=float, default=None,
                         help="PD velocity gain (N*m*s/rad). Default 1 (3 for --walk-style crawl).")
    args = parser.parse_args()
    crawl = args.walk and args.walk_style == "crawl"
    if args.kp is None:
        args.kp = 150.0 if crawl else 80.0
    if args.kd is None:
        args.kd = 3.0 if crawl else 1.0

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    actuator_names = [n.rsplit("_joint", 1)[0] for n in joint_names]
    qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
    qvel_adr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
    gyro_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "angular_velocity")]

    target_pos = DEFAULT_JOINT_POS.copy()
    if args.test_hip is not None:
        target_pos[0] += args.test_hip   # FR_hip
        target_pos[3] += args.test_hip   # FL_hip
        target_pos[6] += args.test_hip   # RR_hip
        target_pos[9] += args.test_hip   # RL_hip
        print(f"TEST MODE: commanding all 4 hip joints to default {args.test_hip:+.2f} rad")
    if args.walk:
        if crawl:
            print(f"WALK MODE: scripted crawl gait with body shift, no RL -- period={args.crawl_period}s, "
                  f"speed={args.crawl_speed}m/s, swing_height={args.crawl_swing_height}m, "
                  f"kp={args.kp}, kd={args.kd}")
        else:
            print(f"WALK MODE: scripted IK {args.walk_style} gait, no RL -- "
                  f"gait_period={args.walk_gait_period}s, stride_length={args.stride_length}m, "
                  f"swing_height={args.swing_height}m")

    n_steps = int(args.seconds / model.opt.timestep)
    kp, kd = args.kp, args.kd
    sim_time = 0.0

    def step():
        nonlocal sim_time, target_pos
        if crawl:
            target_pos = crawl_targets(sim_time, T=args.crawl_period, v=args.crawl_speed,
                                        swing_height=args.crawl_swing_height)
        elif args.walk:
            target_pos = trot_joint_targets(sim_time, args.walk_gait_period, args.stride_length,
                                             args.swing_height, args.stand_depth, args.walk_style)
            if args.roll_gain != 0.0:
                # Active lateral balance: without this, hip stays rigidly at 0 and the robot
                # has no way to counteract the side-to-side rocking a diagonal-support trot
                # naturally induces (confirmed empirically: backward drift was identical
                # regardless of stride length, meaning it wasn't from the propulsion mechanism
                # at all -- it was toppling). Sign is a guess verified the same way as the
                # calf/hip conventions earlier: check the result, flip --roll-gain's sign if
                # the correction makes rocking WORSE instead of better.
                roll = get_roll(data.qpos[3:7])
                roll_rate = data.sensordata[gyro_adr]  # gyro's x-component = roll rate (body frame)
                correction = np.clip(
                    args.roll_gain * roll + args.roll_rate_gain * roll_rate,
                    -args.max_roll_correction, args.max_roll_correction
                )
                target_pos = target_pos.copy()
                target_pos[0] += correction   # FR hip (right)
                target_pos[6] += correction   # RR hip (right)
                target_pos[3] -= correction   # FL hip (left)
                target_pos[9] -= correction   # RL hip (left)
        q = data.qpos[qpos_adr]
        dq = data.qvel[qvel_adr]
        torque = kp * (target_pos - q) - kd * dq
        data.ctrl[act_ids] = torque
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep

    if args.record:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []
        frame_every = max(1, int(round((1.0 / 30.0) / model.opt.timestep)))  # ~30 fps
        start_x = data.qpos[0]
        foot_site_names = ["FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"]
        foot_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in foot_site_names]
        foot_min_z = np.full(4, np.inf)
        foot_max_z = np.full(4, -np.inf)
        for i in range(n_steps):
            step()
            foot_z = np.array([data.site_xpos[sid][2] for sid in foot_site_ids])
            foot_min_z = np.minimum(foot_min_z, foot_z)
            foot_max_z = np.maximum(foot_max_z, foot_z)
            if i % frame_every == 0:
                renderer.update_scene(data, camera="track")
                frames.append(Image.fromarray(renderer.render()))
        frames[0].save(
            args.record, save_all=True, append_images=frames[1:],
            duration=1000 / 30, loop=0
        )
        print(f"Saved {len(frames)} frames to {args.record}")
        final_height = data.qpos[2]
        print(f"Final trunk height = {final_height:.3f} m "
              f"({'OK, standing' if final_height > 0.20 else 'WARNING, may have fallen'})")
        if args.walk:
            net_forward = data.qpos[0] - start_x
            avg_speed = net_forward / args.seconds
            print(f"Net forward displacement: {net_forward:+.3f} m over {args.seconds}s "
                  f"(avg speed {avg_speed:+.3f} m/s)")
            print("\nActual foot height above ground over the full run (world Z, absolute --"
                  " this is what actually matters for ground clearance, not the relative joint "
                  "angle tracking above):")
            for name, zmin, zmax in zip(foot_site_names, foot_min_z, foot_max_z):
                lift = zmax - zmin
                print(f"  {name:16s} min_z={zmin:+.4f}  max_z={zmax:+.4f}  "
                      f"total_lift={lift:.4f} m {'(barely lifts -- likely dragging)' if lift < 0.015 else ''}")
        print_joint_report(data, qpos_adr, joint_names, target_pos, model=model)
    elif args.view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            track_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = track_id
            for _ in range(n_steps):
                step()
                viewer.sync()
    else:
        heights = []
        for i in range(n_steps):
            step()
            if i % int(1.0 / model.opt.timestep) == 0:
                heights.append(data.qpos[2])
        print("Trunk height (m) sampled once per second:", [f"{h:.3f}" for h in heights])
        final_height = data.qpos[2]
        if final_height > 0.20:
            print(f"OK: robot is standing, final trunk height = {final_height:.3f} m")
        else:
            print(f"WARNING: robot may have fallen, final trunk height = {final_height:.3f} m")


if __name__ == "__main__":
    main()

