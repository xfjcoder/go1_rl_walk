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

## Status: step 2 of the staged plan done (verified against our own model). Not yet started: step 3 (defining the RL-residual interface) onward.
