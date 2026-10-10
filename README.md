# Quadruped Locomotion: Unitree Go1 / Go2, MuJoCo + PPO

Trains Unitree Go1 and Go2 quadrupeds to walk — forward, backward, and
sideways — across flat ground, rough terrain, slopes, and stairs, using
MuJoCo physics simulation and PPO (Stable-Baselines3). Nine stages of
development, each building on the last, with every reward-shaping decision,
bug, and failed experiment documented against real logged numbers, not
guesses — see [HISTORY.md](HISTORY.md) for the full story.

<p align="center">
  <img src="media/go2_speed_curriculum_0.8ms.gif" width="300" alt="Go2 walking forward at 0.8 m/s">
  <img src="media/go2_gaitclock_stairs_desc12cm_FIXED.gif" width="300" alt="Go2 descending 12cm stairs cleanly">
  <img src="media/go2_scratch_consolidate_fwd08_rightstrafe03.gif" width="300" alt="Go2 walking forward and strafing sideways simultaneously">
</p>

*Left to right: forward walking at 0.8 m/s; cleanly descending 12cm stairs
(the project's one genuine fix for a problem that resisted four other
approaches); walking forward and sideways simultaneously.*

## Current best checkpoints

Two adopted Go2 checkpoints, both committed and ready to run with no
training — they're genuine specializations, not a strict improvement
either way:

- **`pretrained/go2_gaitclock`** — forward-only, the most thoroughly
  refined: 0% falls across flat ground, rough terrain (0-12cm), and
  slopes (±20°) at 0.3-1.0 m/s, plus an event-driven gait clock that's
  this project's one real fix for the long-standing descending-12cm-stairs
  problem (four earlier reward-shaping approaches all failed at it first).
- **`pretrained/go2_latback`** — adds sideways (strafe) and backward
  walking on top of everything `go2_gaitclock` does, trained fully from
  scratch rather than fine-tuned: 0% falls on flat ground across all 12
  forward×lateral speed combinations, handles rough terrain/slopes/stairs
  well, with two accepted known limits (steep — -20°  — downhill descent,
  and -12cm descending stairs, both specifically at real forward speed).

```bash
pip install -r requirements.txt
python play.py --run-dir pretrained/go2_gaitclock --target-speed 0.8 --record out.gif
python play.py --run-dir pretrained/go2_latback --target-speed 0.0 --target-lateral-speed 0.3 --record out.gif
python eval_policy.py --run-dir pretrained/go2_gaitclock --episodes 16   # fall rate / speed / drift
```

(Earlier checkpoints — `pretrained/go2_terrain`, `go2_hardmine`,
`go2_slopes`, `go2_stairs_hardmine`, `go2_speed_curriculum` — and the
original Go1 lineage — `pretrained/p_stairs`, `o_slope_consolidate`,
`k_hardmine`, `x_mesh_finetune` — are also committed, in case you want a
specific intermediate stage rather than the current best. The
latency-robustness specialization —
`pretrained/go2_latency_teacher_smooth`/`go2_latency_student_smooth` — is
a third adopted-but-specialized checkpoint, trading some stairs capability
for robustness to sim-to-real control latency.)

## The project, in brief

Nine stages, each building on the last (full detail, including every
failure and fix, in [HISTORY.md](HISTORY.md)):

0. **Scripted crawl** (no RL) — a hand-designed gait, to confirm the model
   and physics can walk at all before trusting RL to discover one.
1. **RL, flat terrain** (Go1) — PPO learns to track a commanded forward
   speed from 0.2-1.0 m/s with a clean, symmetric trot.
2. **RL, rough terrain** (Go1) — a random heightfield up to ±12cm, plus
   friction/mass/push domain randomization.
3. **RL, slopes, stairs, discrete obstacles** (Go1) — ramp slopes (±20°)
   and stairs (risers up to 12cm) on top of everything above; discrete
   obstacles needed no dedicated training at all.
4. **Higher-fidelity mesh model** (Go1) — swapped in the official MuJoCo
   Menagerie model (real meshes/inertials), verified the existing
   checkpoint transfers with no retraining.
5. **Higher top speed / flight-phase gait** (Go1, known limitation) — a
   real ceiling found in the always-≥2-feet-down trot/bound gait clock; a
   flight-phase mechanism was built and reaches 1.4 m/s, but with an
   unresolved drift/quality trade-off, accepted as a known limit.
6. **A different robot: Unitree Go2** — the entire recipe (reward shaping,
   PPO setup, curriculum) transferred to a genuinely different quadruped
   with no code changes and no retuning, through rough terrain, slopes,
   and stairs, plus a teacher/student latency-distillation pipeline for
   sim-to-real robustness.
7. **Rangefinder lidar + an event-driven gait clock** (Go2) — lidar gave a
   net regression (tried warm-started and from-scratch, both negative at
   the hardest corner) and was not adopted; the event-driven gait clock
   (closed-loop, pauses the gait phase when a leg is stuck rather than
   advancing on a fixed schedule) was the one approach that actually fixed
   the descending-stairs problem, and is adopted in `go2_gaitclock`.
8. **Sideways and backward walking** (Go2) — trained fully from scratch to
   test whether that avoids fine-tuning's "shared-network interference";
   confirmed a left/right strafe asymmetry found during this work is
   stochastic (training-path-dependent), not a fixed structural property
   of the robot, by reproducing it in the opposite direction across two
   independent training lineages. Adopted as `go2_latback`.
9. **Onboard depth camera** (Go2) — tested whether real visual perception
   helps with the never-solved descending-stairs problem, given lidar
   already didn't; stopped early once a direct check showed the camera's
   information content at this configuration may not be meaningfully
   richer than lidar's sparse rays for that specific problem. Not adopted,
   infrastructure kept for a possible better-scoped retry.
10. **Point-to-goal navigation** (Go2) — a new direction on top of walking:
    drive `go2_latback` toward an arbitrary world-frame goal with a thin
    proportional outer loop and **zero retraining**, exploiting the fact
    that the walking reward already holds heading fixed. 94-100% success,
    0% falls across flat and rough terrain (32 randomized goals each).
    Extended with static obstacle avoidance (a simplified potential field):
    0% collisions even where an obstacle sat directly on the path, with a
    well-understood (not chased further) convergence weak spot when a goal
    happens to land near an obstacle. Further extended with multi-waypoint
    patrol routes (`--waypoint`/`--loop`), reusing the same steering and
    avoidance logic unchanged, and with dynamic (moving) obstacles
    (`--nav-obstacle-speed`), which surfaced a real, measured limitation:
    collision rate rises from 0% to 16% once an obstacle can wander into a
    route that started clear, even though success rate and fall rate stay
    unchanged. Finally extended with perception-based obstacle detection
    (`--lidar-nav`), repurposing an onboard 9-ray rangefinder fan built in
    an earlier stage for the walking policy's own observation, now reused
    by the navigation layer instead of privileged ground-truth positions.
    A short-range, forward-cone-only sensor, as measured directly (~1m,
    dropping off to the sides) — yet randomized trials found it matched
    or slightly *beat* ground-truth avoidance overall, because the
    existing potential-field steering law's own weakness (confused by
    several simultaneous obstacles) hurt more than the sensor's limited
    range did, a more nuanced result than expected going in. Real turning
    capability was then
    attempted (training the policy itself, not just the outer loop) and
    found a genuine, repeatable limit: four structurally different
    training approaches all converged on the same failure — the robot
    holds a small bounded heading wobble instead of sustaining a turn.
    Research context suggests this is specific to this project's compute
    scale, not a fundamental limit of RL — others have solved the same
    problem with the same algorithm (PPO) using orders of magnitude more
    training experience via massively-parallel GPU simulation. Two real
    bugs were found and fixed along the way (one general, project-
    wide lesson about `--resume`'s learning-rate schedule); not adopted.

## Repository layout

```
go1_rl_walk/
├── assets/                 # MJCF models: go1.xml / go1_mesh.xml (Go1), go2_mesh.xml (Go2,
│                            #   primary focus), go2_mesh_lidar.xml / go2_mesh_camera.xml
│                            #   (opt-in sensor variants, see Stages 7/9 in HISTORY.md)
├── envs/go1_env.py         # Gymnasium env: obs, action, reward, termination, terrain (both robots)
├── scripted_gait.py        # Hand-designed crawl gait (stage 0, no RL)
├── train.py                # PPO training entrypoint (--run-name writes to runs/<name>/)
├── eval_policy.py          # Randomized multi-episode eval: fall rate, speed, drift
├── gait_stats.py           # Per-foot diagnostics: step rate, duty, swing height
├── play.py                 # Load a checkpoint (--run-dir) and watch it walk
├── navigate.py             # Drive a checkpoint toward a world-frame (x, y) goal, no retraining
├── eval_navigate.py        # Randomized multi-episode nav eval: success rate, time-to-goal, path efficiency
├── distill_student.py      # Behavior-cloning distillation (teacher -> latency-robust student)
├── expand_obs_checkpoint.py # Warm-start a larger-observation policy from a smaller one
├── smoke_test.py           # Sanity-check the model with no RL deps; also runs the crawl gait
├── runs/<name>/            # One dir per training run: checkpoints/, logs/, args.json, env_kwargs.json
│                           #   (gitignored -- see pretrained/ for committed checkpoints)
├── pretrained/              # Committed, ready-to-run checkpoints -- each a valid --run-dir,
│   ├── go2_gaitclock/       #   no training required. go2_gaitclock / go2_latback are current
│   ├── go2_latback/         #   best (see above); the rest are earlier stages or specializations.
│   └── ...
├── requirements.txt
└── LICENSE                 # MIT
```

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate   # or conda
pip install -r requirements.txt
```

GPU is optional for MuJoCo (CPU physics is fine), but strongly speeds up
PPO's neural-net updates. This machine trains at ~800-2800 steps/s on CPU
alone (higher with more `--n-envs` and once resuming a converged policy,
since episodes run to their full 1000-step length instead of ending early).

## 2. Sanity check the model first

```bash
cd go1_rl_walk
python smoke_test.py --view
```

This loads `assets/go1.xml`, holds the standing pose with a simple PD
controller (no RL yet), and opens the MuJoCo viewer so you can confirm
the robot stands without exploding/clipping through the floor.

Then check it can actually walk, still with no RL, using the scripted crawl
gait in `scripted_gait.py` (one leg swings at a time, with a body-weight
shift onto the other three so the center of mass never leaves the support
triangle — the earlier scripted attempts skipped the weight shift and went
nowhere, or worse, toppled):

```bash
python smoke_test.py --walk --record walk_crawl.gif --seconds 30
```

Confirming this before spending compute on training separates "the model/
physics can't walk" from "RL hasn't learned to walk yet" — the two look
identical from a failed training run alone.

> **Viewer crashes with a GLFW/Wayland segfault?** This is a known
> GLFW-on-Wayland issue, not a project bug. Try forcing XWayland:
> ```bash
> WAYLAND_DISPLAY= python3 smoke_test.py --view
> ```
> If that doesn't work, skip the interactive viewer entirely and render
> offscreen to a GIF instead (no window/GLFW involved at all):
> ```bash
> python3 smoke_test.py --record stand.gif
> ```
> `play.py` supports the same `--record out.gif` flag for watching a
> trained policy without the interactive viewer.

> **Model note:** `assets/go1.xml` is built from published Go1 reference
> dimensions (trunk box, leg offsets, link lengths, joint ranges, motor
> torque limits — 23.7 N·m hip/thigh, 35.55 N·m knee) using primitive
> capsule/box geometry, so it's dynamically reasonable and needs no
> external mesh files. For visual fidelity or exact inertial tuning,
> swap in the official model from MuJoCo Menagerie once you have
> internet access:
> ```bash
> git clone https://github.com/google-deepmind/mujoco_menagerie
> ```
> then point `DEFAULT_XML` in `envs/go1_env.py` at
> `mujoco_menagerie/unitree_go1/go1.xml`. The env addresses joints,
> actuators, and sensors by name, so either file works unmodified.

## 3. Evaluate and watch a trained policy

```bash
python eval_policy.py --run-dir pretrained/go2_gaitclock --episodes 16 --seconds 20
```
Runs many randomized episodes (reset jitter, friction, and — for terrain runs
— mass/terrain/pushes) in parallel and reports fall rate, speed tracking, and
drift, sweeping the trained command/terrain range automatically. **Judge any
change this way, not from a single episode** — single scripted-gait runs
early in this project swung between "walks 0.5 m" and "falls over" for
neighboring parameters, purely from randomness, and this lesson recurred
throughout — a single favorable (or unfavorable) seed has misled conclusions
more than once, see [HISTORY.md](HISTORY.md).

```bash
python gait_stats.py --run-dir pretrained/go2_gaitclock --target-speed 0.8 --stair-height -0.12
```
Per-foot step rate, duty cycle, and swing height/timing — catches a lopsided
gait (e.g. one leg stepping much more/less often than the others, or
visibly smaller strides) that fall rate and speed alone hide.

```bash
python play.py --run-dir pretrained/go2_gaitclock --target-speed 0.8 --stair-height -0.12 --record out.gif
```
Watch it (or record a GIF, avoiding the GLFW viewer entirely). `--run-dir`
rebuilds the exact env the checkpoint was trained with (kp/kd, reward
weights) from that run's saved `env_kwargs.json`. `pretrained/go2_latback`
also accepts `--target-lateral-speed` for sideways/backward commands.

## How the task is set up

- **Action space**: 12 target joint-position offsets (one per hip/thigh/
  calf joint), PD-tracked to torque at 250 Hz internally.
- **Observation** (51-dim base; `pretrained/go2_latback` uses the same
  51 dims, just with the lateral-speed slot populated): gravity vector in
  body frame, base angular velocity, base orientation quaternion, 12 joint
  positions (relative to a neutral standing pose), 12 joint velocities,
  the previous action, a 3-dim velocity command (`[target_speed,
  target_lateral_speed, 0]` — the third slot, yaw-rate, is reserved but
  unused), and a 2-dim sin/cos gait-phase clock. Both adopted checkpoints
  are **blind** — no terrain information (heightmap, ray casts, camera) —
  they only feel the ground through joint torques and the trunk IMU.
  Optional terrain-aware sensors (an analytic local heightmap, a
  rangefinder lidar fan, and an onboard depth camera) were all tried as
  separate experiments (Stages 3d, 7, 9 in [HISTORY.md](HISTORY.md)) but
  none ended up in either adopted checkpoint.
- **Reward**: forward/lateral-velocity tracking (body-frame, if
  `--body-frame-velocity`) + upright orientation + heading/lateral-position
  drift penalties + torque/energy cost + action-rate smoothness + a
  diagonal-trot timing bonus against a speed-scaled clock
  (`--phase-match-weight`, optionally relaxed on steep stairs/slopes via
  `--phase-match-stair-relax`/`--phase-match-slope-relax`) + a capped foot
  air-time bonus + a foot-clearance bonus (its target scales up
  per-episode to clear the current stair height,
  `max(target_clearance, stair_h+0.03)`) + survival bonus. Optional
  per-foot duty-cycle penalties and a trot-symmetry bonus exist but are
  used differently across checkpoints — see `train.py --help` and
  [HISTORY.md](HISTORY.md) for why. Weights are in
  `Go1FlatEnv._compute_reward`.
- **Termination**: trunk tips past ~60° from vertical, or trunk height
  (relative to the local terrain) leaves the `[0.15, 0.6]` m band.
- **Foot contact**: real MuJoCo contact force (summed over all of a foot's
  contact points, >1 N), not a height threshold — a height threshold is
  wrong the moment the ground isn't flat, and was actually wrong on flat
  ground too in an earlier version (the foot site sits at the sphere's
  center, above its resting height, so a standing robot was seen as having
  zero feet down).
- **Domain randomization**: joint/height/yaw jitter and floor+foot friction
  on every run; trunk mass scale and random horizontal pushes on terrain
  runs (`--mass-scale-range`, `--push-velocity`).
- **Rough terrain, slopes, stairs** (`--terrain-amp-max`/`--slope-max-deg`/
  `--stair-height-max`): all three share one MuJoCo heightfield, regenerated
  every episode and additive (any combination can be nonzero at once — a
  sloped, bumpy staircase). Bumps are interpolated random noise (feature
  size 15 cm-1 m); a slope is a flat pad, then a constant grade over
  `--ramp-length`, then a flat plateau; stairs are the same shape with a
  step function instead. Direction (uphill/downhill, ascending/descending)
  is represented by which end of the pad is elevated, not a negative
  height — a heightfield can't go below its own z=0 plane, and each can be
  biased toward one direction specifically for hard-mining
  (`--stair-ascending-prob`/`--slope-uphill-prob`). A stair riser is a
  steep ramp within one 5 cm grid cell, not a true vertical face. The
  terrain-generating code is only built into the model when at least one
  of these is enabled — the flat env is bit-identical to before any
  terrain support existed.
- **Gait clock timing** (`--gait-period-fast`/`--gait-period-stair-stretch`):
  the trot clock's period shortens for higher commanded speed and
  lengthens for a taller current-episode stair, giving a big lift more
  real time to complete instead of being rushed by a fixed rhythm. An
  event-driven variant (`--gait-clock-wait-for-contact`) pauses the clock
  entirely when a leg is overdue for its prescribed stance contact,
  instead of always advancing on a fixed schedule — the mechanism behind
  `go2_gaitclock`'s descending-stairs fix (Stage 7).

## Troubleshooting

- **Robot immediately collapses in smoke_test.py**: check `kp`/`kd` gains
  and the `stand` keyframe joint angles match your intended standing pose;
  also confirm the floor geom has nonzero friction.
- **Scripted crawl gait (`--walk`) goes nowhere or falls over**: check the
  robot's center of mass against the support triangle of the feet actually
  on the ground during each swing — a gait that doesn't shift body weight
  onto the remaining stance feet before lifting one will have the CoM
  outside the triangle for some swings (this is exactly what made the
  earliest scripted gait here move ~0 m net despite "walking" motion).
- **Policy learns to stand but never walks**: increase the velocity-
  tracking weight relative to the survival bonus, or reduce the survival
  bonus — it's easy for a lazy "just stand still" policy to dominate early
  training, especially at a low target speed where standing still already
  scores most of the tracking reward (this is why the reward normalizes
  velocity error by `target_speed` rather than using an absolute value).
- **Policy learns to lunge and fall rather than walk**: check `train/std`
  early in training. If gSDE is on and the effective action noise is large,
  the untrained policy may never survive long enough to discover standing,
  let alone walking — see the gSDE note above. `--no-use-sde` (the default)
  with `--log-std-init` around -1.0 to -1.5 avoids this.
- **Asymmetric/lopsided gait** (e.g. front legs stepping twice as often as
  rear, or one diagonal pair almost always down and the other almost always
  up): check `gait_stats.py`'s per-foot step rate and duty cycle. Likely
  causes, roughly in the order to check: an uncapped `--air-time-weight`
  (add `--air-time-cap`), a `--phase-match-weight` of 0 (a canonical trot
  needs the clock, not just the contact-count/duty terms), or a `--trot-
  weight` that's been gamed (prefer the clock over this).
- **Policy degrades into near-random motion after a resume** (e.g. only one
  side of the robot moves, legs drag): check `train/std`. If it's climbed
  well above ~1.0 (action space is `[-1, 1]`, so std should normally stay
  under ~0.5), this is **entropy blowup**: the `ent_coef` entropy bonus for
  continuous actions is unbounded above, and if the policy-gradient signal
  goes quiet — which commonly happens right after changing a reward term
  mid-resume, since the value function needs time to re-adapt — the entropy
  term can dominate the loss and drive `std` upward without limit.
  `train.py`'s `StdGuardCallback` auto-stops training if `std` exceeds 1.5,
  so this costs a few thousand wasted steps instead of an entire run. Avoid
  it by changing `--ent-coef` and a reward-shaping weight in separate runs,
  not the same resume.
- **A resumed run trains fine but the resulting policy acts worse than
  before resuming**: check that `--resume` found matching VecNormalize
  stats (`train.py` prints `Loaded VecNormalize stats from ...`, or a
  `WARNING` if it couldn't) — a fresh observation normalizer feeds a
  resumed policy differently-scaled inputs until its running stats catch
  up, which can look like the policy itself regressed.
- **A good checkpoint got silently overwritten by a later bad run**: use
  `--run-name` — every run then gets its own `runs/<name>/checkpoints/`
  instead of writing to the shared `checkpoints/`. Within a run, the
  numbered `go1_flat_<steps>_steps.zip` checkpoints (every ~1M steps) and
  the stamped `go1_flat_final_<total_steps>.zip` are never overwritten;
  only the unstamped `go1_flat_final.zip` is, on a second run in the same
  directory.
- **Rough terrain causes huge contact forces or an immediate flip on
  reset**: make sure the spawn logic that lifts the trunk above the local
  terrain height actually ran (it's automatic whenever `terrain_amplitude`
  is set) — a foot spawning even slightly below a heightfield surface (as
  opposed to a flat plane, which tolerates this gently) can produce contact
  forces in the kilonewtons and toss the robot instantly.
- **Randomizing `--friction-range` seems to do nothing on terrain runs**:
  the foot geoms have `priority="1"` in `assets/go1.xml`, so their own
  friction value overrides the floor/terrain geom's in every contact —
  randomizing only the floor/terrain never touches what the feet actually
  feel. `Go1FlatEnv` randomizes the foot geoms too, but only when terrain
  is enabled (flat runs keep the old floor-only behavior for reproducibility).
- **Robot walks out of the camera frame**: the model includes a `track`
  camera (mode `trackcom`) that follows the trunk — both the interactive
  viewer and `--record` use it by default.
- **Training is slow**: increase `--n-envs` (bounded by CPU cores), or
  reduce `n_steps`/`batch_size` for faster iteration at the cost of some
  sample efficiency. Episodes that end early (falls) are also much faster
  per step than full-length ones, so steps/s rising over a run usually
  means the policy is surviving longer, not that something got faster.
