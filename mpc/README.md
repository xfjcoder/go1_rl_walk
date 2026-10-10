# Stage 11: MPC exploration (evaluation track, not yet adopted)

Investigates whether a conventional MPC-based locomotion controller -- rather than
training a new RL policy from scratch -- can give this project genuine turning
capability, something 4 different from-scratch RL approaches in Stage 10c could not
achieve (see `HISTORY.md`'s Stage 10c section). The idea: use MPC as the locomotion
base (it naturally supports yaw-rate commands via footstep/gait scheduling), with RL as
a later residual correction on top, if this track is adopted.

This is a **separate evaluation track**, isolated on its own branch (`stage11-mpc`) and
its own Python environment. Nothing here touches `envs/go1_env.py`'s own RL pipeline or
`assets/go2_mesh.xml`.

## What's here

- `third_party/go2-convex-mpc` (git submodule): a third-party convex MPC controller for
  the Unitree Go2, built in MuJoCo (MIT license, UC Berkeley MEng capstone project,
  https://github.com/elijah-waichong-chan/go2-convex-mpc). Picked over the more mature
  `iit-DLSLab/Quadruped-PyMPC` (527 stars, peer-reviewed, real-hardware tested) because
  its install is dramatically simpler in this sandbox -- Quadruped-PyMPC requires
  compiling acados from source; this one is `pip install -e .`.
- `go2_mesh_for_mpc.xml`: a cosmetically-adapted copy of `../assets/go2_mesh.xml` (see
  that file's own header comment for exactly what changed and why -- relative meshdir,
  body renamed to `base_link`, leg declaration order matched to the controller's
  convention). Verified byte-identical mass/inertia/joint-limits/actuator-limits to our
  own `go2_mesh.xml` before any of this was attempted.
- `verify_turning.py`: runs the submodule's own command-scheduled walk+turn demo, but
  driving **this project's own Go2 model** instead of the submodule's bundled one.
  Headless (no interactive viewer -- this sandbox has no display, so
  `mujoco.viewer.launch_passive` isn't usable; same constraint `navigate.py`'s own
  `render_mode="human"` path runs into). Saves a trajectory plot and an offscreen-
  rendered GIF to `media/`.

## Setup (separate venv -- do NOT use this project's own `venv`)

The submodule pins `numpy<2` and `requires-python <3.11`; this project's own venv uses
numpy 2.x / Python 3.12. Keep them apart:

```bash
python3 -m venv mpc/venv
source mpc/venv/bin/activate
pip install -r mpc/requirements.txt
pip install -e third_party/go2-convex-mpc --ignore-requires-python
```

The `--ignore-requires-python` bypass is safe here: the `<3.11` pin turned out to be
inherited packaging metadata, not an actual code requirement -- verified working on
3.12. `numpy<2` is a real constraint though: scipy/matplotlib/pinocchio's *current*
releases require numpy>=2, so `pip install -r requirements.txt` must land on compatible
older releases of those three specifically (confirmed: numpy 1.26.4, scipy 1.12.0,
matplotlib 3.8.4, pin 4.1.0 all import and run correctly together, despite pip printing
a non-fatal dependency-conflict warning about `cmeel-boost` wanting numpy>=2 -- that
warning turned out to be about an unused transitive pin, not a real runtime conflict).

## Run

```bash
source mpc/venv/bin/activate
python3 mpc/verify_turning.py
```

## What was found

1. **The XML swap is physically identical** -- byte-identical mass, inertia, joint
   ranges, actuator torque limits to `assets/go2_mesh.xml` (3-seed check with nonzero
   actions). Only a friction difference in the *submodule's own* ground plane (0.4 vs.
   our project's 0.9 sliding friction) -- a world parameter, not a robot one.

2. **A real integration bug, not a model incompatibility**: `mujoco_model.py`'s
   `update_pin_with_mujoco()` copies MuJoCo's `qpos[7:]` into the controller's Pinocchio
   model *positionally* -- no name-based remapping. This project's own `go2_mesh.xml`
   declares legs in a different order (FR,FL,RR,RL) than the submodule's bundled model
   (FL,FR,RL,RR). Swapping in our XML without fixing this silently swapped which leg's
   data went where -- caught via a foot-position sanity check (one foot landed at a
   wildly wrong position) before it could produce a misleading result. Fixed by
   reordering the four leg `<body>` subtrees in `go2_mesh_for_mpc.xml` to match the
   submodule's convention (XML declaration order doesn't affect physics, only array
   indexing -- confirmed mass/hierarchy unchanged after reordering).

3. **Genuine, repeatable turning, verified against our own robot model**: the
   submodule's own 10-second command-scheduled demo (forward / sideways / stop / pure
   in-place rotation at 2.0 rad/s / combined forward+rotation / forward) was run against
   `go2_mesh_for_mpc.xml` end to end (2000/2000 control ticks, no failures). Achieved
   yaw rate: **1.91 rad/s** during the pure-rotation phase and **1.93 rad/s** during the
   combined walk+turn phase, against a 2.0 rad/s command -- both *tighter* than the
   submodule's own bundled model achieved in the same test (1.80 / 1.85 rad/s). No
   spurious rotation during straight-walking phases (<0.01 rad/s). This is the genuine,
   controlled, simultaneous translation-and-rotation capability that 4 different
   from-scratch RL training attempts in Stage 10c could never sustain.

See `media/mpc_turning_demo.gif` and `media/mpc_turning_summary.png`.

## Step 3: the RL-residual interface

`mpc_residual_env.py` wraps `CentroidalMPC` + `LegController` (unmodified, imported
straight from the submodule) in a Gymnasium `Env`. Each control tick, MPC computes its
usual nominal 12-dim joint torque command exactly as in `verify_turning.py`; an RL
action (`Box(-1,1,shape=(12,))`, scaled by `residual_scale` N·m) is added to it before
clipping to the actuator limits. MPC still does 100% of the locomotion planning
(footstep timing, swing trajectories, stance force allocation, turning) -- nothing
about its own logic is touched; RL is a pure bolt-on correction.

- **Observation** (47-dim): body-frame linear velocity (3), body-frame angular velocity
  (3), roll/pitch (2), joint positions (12), joint velocities (12), the MPC's own
  nominal torque for that tick (12, giving the policy context on what the base
  controller already decided), and the current command -- target forward/lateral
  speed and yaw rate (3).
- **Reward**: survival (+1/tick) + Gaussian tracking reward on forward speed, lateral
  speed, and yaw rate (`exp(-(error/sigma)^2)`, same functional form `envs/go1_env.py`
  already uses) + a penalty on residual magnitude (`-residual_effort_weight * sum(residual^2)`,
  encouraging the policy to intervene minimally rather than override the base
  controller). Terminates on a fall (base height or tilt threshold).
- **Domain randomization**: floor friction and base mass scale sampled each reset,
  optional random push perturbations -- mirrors `envs/go1_env.py`'s own conventions, so
  the residual's job is explicitly "handle what the MPC's rigid-body model doesn't
  plan for," matching how every other RL stage in this project has been scoped.

**Verification** (`verify_residual_interface.py`): with the residual held at exactly
zero every step (command forced to the same 2.0 rad/s in-place rotation
`verify_turning.py`'s own test used), the env ran stably for the full episode (no
falls) and tracked the command in the right direction and approximately the right
magnitude: **1.83-1.84 rad/s achieved**, consistently across the whole run (checked in
0.5s windows -- not a transient that converges toward the reference value over time).

This is close to but not an exact match for the standalone script's own **1.91 rad/s**
result for the same command. I checked two specific hypotheses for the gap and ruled
both out directly rather than assuming:
1. *Measurement convention* -- `verify_turning.py` reports Pinocchio's centroidal
   angular velocity, this env's observation uses MuJoCo's raw base-body angular
   velocity (`qvel[3:6]`). Compared both directly in the same run: 1.8349 vs 1.8325
   rad/s -- effectively identical, not the cause.
2. *Pre-loop trajectory-generator initialization* -- the standalone script's very
   first (pre-loop) `traj.generate_traj()` call always uses a zero command, this env's
   `reset()` used the actual target command for that same call. Patched the env to
   match the original's zero-command initialization exactly: no change (1.838 rad/s).

Neither hypothesis explains it; the remaining likely candidates are OSQP's own
warm-start/numerical sensitivity across separate process runs, or some other
initialization-order detail not yet isolated. Reported honestly rather than claimed
fixed -- this is a ~4% gap, not evidence the residual mechanism itself is broken (the
interface reproduces stable, correctly-directioned, closely-tracking-magnitude
behavior with zero intervention, which is the actual correctness bar for this step).

## Status: steps 1-3 of the staged plan done. Not yet started: step 4 (train the residual policy, same randomized multi-seed evaluation discipline as every other RL stage in this project) and step 5 (decide: adopt or document as a negative result, same as Stage 10c).
