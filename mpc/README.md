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
falls) and tracked the command well: **1.955 rad/s achieved** against the 2.0 rad/s
command, consistent with (even marginally exceeding) the standalone script's own
1.91 rad/s result for the identical test.

That number is from AFTER a self-review caught two real issues in the first version of
this interface, worth recording since the first version's zero-residual result (1.83-
1.84 rad/s) looked plausible enough that it could have gone unnoticed:

1. **A real bug: `xml_path` was silently ignored.** `MuJoCo_GO2_Model()`'s constructor
   hardcodes a *module-level* `XML_PATH` global -- it isn't a constructor argument. The
   first version stored `self.xml_path` but never applied the
   `mujoco_model_module.XML_PATH = ...` monkeypatch that `verify_turning.py` correctly
   does, so the env was silently driving the submodule's own bundled model (confirmed:
   its loaded model name was `"go2 scene"`, not `"go2_mesh"`) the whole time. This
   fully explains the ~4% gap reported in an earlier version of this doc -- it wasn't a
   mysterious OSQP sensitivity, it was the wrong robot model.
2. **A severe performance bug: `reset()` cost ~2.4 seconds.** Timed each component
   directly: `PinGo2Model()` and `ComTraj()` each cost ~1.2s on *every* instantiation
   (not a one-time process cost -- confirmed by constructing each twice in the same
   process), while `generate_traj()` and `CentroidalMPC()` are cheap (~3ms, ~6ms). At
   2.4s/reset, real PPO training (needing thousands of episodes) would have been
   impractical. Fixed by constructing `go2`/`traj`/`mujoco_go2`/`leg_controller`/`gait`
   **once**, in `__init__`, and having `reset()` only reset *state* on those same
   objects (`mj_resetData` + `update_with_q_pin` + re-sync Pinocchio + regenerate the
   trajectory for the new episode's command) -- confirmed safe since
   `ComTraj.generate_traj()`'s only coupling to `go2` is reading its current state each
   call, not accumulating hidden state of its own. Result: reset() now costs **~8-10ms**
   (a ~250x speedup), with the one-time ~2.6s cost paid once at env construction instead
   of every episode.

Both fixes are in the committed version of `mpc_residual_env.py` -- the numbers above
are from the corrected environment, not the one with these bugs.

**Still worth a decision before step 4, not bugs**: control rate is 200Hz (matching the
MPC's own leg-controller update rate), four times more steps per episode-second than
this project's usual 50Hz RL control rate -- may be worth decimating RL's own decision
rate for more standard PPO credit-assignment horizons. Fall thresholds (base height
0.15m, tilt 0.9rad) are reasonable guesses, not yet empirically validated against this
specific robot/controller's actual failure modes.

## Step 4: training, and a third real bug

`train_residual.py` trains the residual with PPO (`stable-baselines3`), mirroring
`train.py`'s own hyperparameters and conventions where they transfer directly (linear LR
decay, `VecNormalize`, `CheckpointCallback`, `RewardComponentLoggingCallback` reused
unmodified, per-run `args.json`/`env_kwargs.json`). `SubprocVecEnv` uses
`start_method="spawn"`, not Linux's default `fork` -- CasADi/Pinocchio/OSQP are native
libraries not guaranteed fork-safe.

**A third real bug, found the first time training actually ran for more than one
iteration.** The very first launch (8 parallel envs) completed its first iteration
cleanly, then died with the same `EOFError` symptom as before -- but this time the
underlying cause turned out to be completely different from the first two bugs, and far
more consequential: building a retry wrapper (`run_resilient.sh`, auto-resume from the
latest checkpoint on crash) to work around what looked like intermittent flakiness
instead **made the problem reproduce 100% of the time** -- all 30 retry attempts
crashed, always right after resuming, always within one iteration. That consistency was
the actual clue: resuming from the exact same checkpoint with the exact same seed
produces the exact same action sequence, so if a crash is deterministic given that
sequence, retrying with identical inputs just hits it again. "Add retries" was treating
a symptom; the actual bug needed isolating.

Reproduced directly (`SubprocVecEnv` + `VecNormalize.load` + `PPO.load`, bypassing the
wrapper script entirely) and got the real error for the first time:
```
RuntimeError: Error in Function::call for 'S' [OsqpInterface] ...
conic process failed. Set 'error_on_fail' option to false to ignore this error.
```
OSQP (the QP solver underneath `CentroidalMPC`) sometimes can't find a feasible solution
given the robot's current state. That's a genuine, catchable Python exception -- but
nothing in this env or the submodule catches it, so it propagates up through
`SubprocVecEnv` and kills the worker process outright (the `EOFError` was always just
the symptom one layer up: the parent's pipe read failing because the process on the
other end had already died).

Fixed by catching `RuntimeError` around the `solve_QP` call in
`mpc_residual_env.py::step()`: on failure, keep the last successful solution for that
one tick (a reasonable one-tick-stale fallback) and terminate the episode immediately
with a fixed `-5.0` reward -- the same treatment as a fall, since a state the base
controller itself can't solve for is at least as bad as falling over, and the policy
should learn to avoid causing either one. Re-ran the exact scenario that crashed 30/30
times before the fix: 5 clean iterations, zero crashes. Bonus confirmation the fix
targets the right thing: `ep_len_mean` started at 57 (frequent early QP-failure
terminations) and grew to ~1700-1800 within those same 5 iterations -- the `-5.0`
penalty is already visibly teaching the policy to avoid whatever state triggers it.

`train_residual.py` also gained `--auto-resume` (correctly computing `remaining =
target - checkpoint's own num_timesteps` each time, to avoid the LR-schedule dilution
bug documented in Stage 10c) and checkpoints save every iteration instead of every
200,000 steps, so a crash -- this kind or any other -- now loses at most one iteration's
progress. `run_resilient.sh` wraps it with automatic retry. Both remain in place even
though the actual crash is fixed now: cheap insurance, not a workaround for something
unresolved.

## Status: steps 1-4 of the staged plan done (interface defined and reviewed; training launched, hit and fixed a third real bug: uncaught QP-solver-failure exceptions crashing worker processes). Not yet complete: step 4's actual training run (in progress) and step 5 (decide: adopt or document as a negative result, same as Stage 10c).
