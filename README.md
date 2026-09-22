# Go1 Quadruped Locomotion

Trains a Unitree Go1 to walk with MuJoCo physics + PPO (Stable-Baselines3).
The pipeline has three stages, each building on the last:

0. **Scripted crawl** (no RL) — a hand-designed gait, to check the model and
   physics can walk at all before trusting RL to discover one.
1. **RL, flat terrain** — PPO learns to track a commanded forward speed from
   0.2–1.0 m/s with a clean, symmetric trot.
2. **RL, rough terrain** — fine-tuned on a random heightfield up to ±12 cm,
   plus friction/mass/push randomization.

All three are done. The current best checkpoint is `runs/k_hardmine`: 0%
falls across the whole tested grid (0/4/8/12 cm terrain amplitude × 0.3/0.8/1.0
m/s commands). See [Current status](#current-status) for the full results and
[How this was trained](#how-this-was-trained) to reproduce it from scratch.

```
go1_rl_walk/
├── assets/go1.xml         # MJCF model (self-contained, capsule-based)
├── envs/go1_env.py        # Gymnasium env: obs, action, reward, termination, terrain
├── scripted_gait.py       # Hand-designed crawl gait (stage 0, no RL)
├── configs/flat_terrain.yaml  # original reward-weight defaults (historical reference —
│                          #   the actual tuned recipe is in this README, not this file)
├── train.py               # PPO training entrypoint (--run-name writes to runs/<name>/)
├── eval_policy.py         # Randomized multi-episode eval: fall rate, speed, drift
├── gait_stats.py          # Per-foot diagnostics: step rate, duty, swing height
├── play.py                # Load a checkpoint (--run-dir) and watch it walk
├── smoke_test.py          # Sanity-check the model with no RL deps; also runs the crawl gait
├── runs/<name>/           # One dir per training run: checkpoints/, logs/, args.json, env_kwargs.json
└── requirements.txt
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

## How this was trained

`train.py --help` lists every flag, grouped (run bookkeeping, PPO/exploration,
physics/control, command speed, gait clock, reward shaping, terrain, domain
randomization). Rather than guessing a combination from scratch, this is the
actual staged recipe that produced `runs/k_hardmine`, each stage `--resume`ing
the last. Exact flags for any past run are also always in that run's
`runs/<name>/args.json`.

**Stage 1a — flat, fixed 0.3 m/s, diagonal trot clock** (`runs/h_clock`, 6M
steps, ~15 min):
```bash
python train.py --run-name my_h_clock --timesteps 6000000 --n-envs 8 \
    --target-speed 0.3 --kp 80 --kd 2 --log-std-init -1.0 \
    --air-time-cap --air-time-weight 1.0 --foot-clearance-weight 0.3 \
    --phase-match-weight 0.5 --phase-match-warmup-steps 1000000 \
    --gait-period 0.7 --gait-style trot
```

**Stage 1b — speed curriculum, 0.2-1.0 m/s** (`runs/i_speed_curriculum`, +12M
steps, ~70 min):
```bash
python train.py --run-name my_speed --n-envs 16 --timesteps 12000000 \
    --resume runs/my_h_clock/checkpoints/go1_flat_final_*.zip \
    --resume-log-std -1.6 --learning-rate 2e-4 \
    --speed-range-min 0.2 --speed-range-max 1.0 \
    --speed-curriculum-start 0.3 --speed-curriculum-steps 8000000 \
    --gait-period 0.7 --gait-period-fast 0.45 --gait-style trot \
    --phase-match-weight 0.5 --phase-match-warmup-steps 1000000 \
    --lateral-tracking-weight 0.5 --body-frame-velocity \
    --kp 80 --kd 2 --air-time-cap --air-time-weight 1.0 --foot-clearance-weight 0.3
```

**Stage 2a — rough terrain, 0-12 cm heightfield** (`runs/j_terrain`, +14M
steps, ~1.6 h):
```bash
python train.py --run-name my_terrain --n-envs 16 --timesteps 14000000 \
    --resume runs/my_speed/checkpoints/go1_flat_final_*.zip \
    --resume-log-std -1.8 --learning-rate 2e-4 \
    --terrain-amp-max 0.12 --terrain-curriculum-start 0.04 --terrain-curriculum-steps 8000000 \
    --friction-range 0.5 1.25 --mass-scale-range 0.9 1.1 --push-velocity 0.3 \
    --speed-range-min 0.2 --speed-range-max 1.0 --speed-curriculum-start 1.0 --speed-curriculum-steps 1 \
    --gait-period 0.7 --gait-period-fast 0.45 --gait-style trot \
    --phase-match-weight 0.5 --phase-match-warmup-steps 1 \
    --lateral-tracking-weight 0.5 --body-frame-velocity \
    --kp 80 --kd 2 --air-time-cap --air-time-weight 1.0 --foot-clearance-weight 0.3
```

**Stage 2b — hard-mine the worst corner** (`runs/k_hardmine`, +8M steps, ~55
min): the curriculum above only ramps the terrain amplitude's *upper* bound,
so episodes keep sampling amplitude from `U(0, 12cm)` even late in training —
the truly hard combination (large bumps at high speed together) stays a thin
slice of what the policy ever sees. Fine-tune with the sampling biased
directly at the hard region instead of widening the curriculum further:
```bash
python train.py --run-name my_hardmine --n-envs 16 --timesteps 8000000 \
    --resume runs/my_terrain/checkpoints/go1_flat_final_*.zip \
    --resume-log-std -1.8 --learning-rate 1.5e-4 \
    --terrain-amp-min 0.06 --terrain-amp-max 0.12 --terrain-curriculum-start 0.12 --terrain-curriculum-steps 1 \
    --speed-range-min 0.5 --speed-range-max 1.0 --speed-curriculum-start 1.0 --speed-curriculum-steps 1 \
    --friction-range 0.5 1.25 --mass-scale-range 0.9 1.1 --push-velocity 0.3 \
    --gait-period 0.7 --gait-period-fast 0.45 --gait-style trot \
    --phase-match-weight 0.5 --phase-match-warmup-steps 1 \
    --lateral-tracking-weight 0.5 --body-frame-velocity \
    --kp 80 --kd 2 --air-time-cap --air-time-weight 1.0 --foot-clearance-weight 0.3
```

Monitor any of these with `tensorboard --logdir runs` (`rollout/ep_len_mean`
climbing to 1000 = full-length episodes; `reward_components/*` breaks the
total down by term; `curriculum/*` shows the speed/terrain ramps).

## 3. Evaluate and watch a trained policy

```bash
python eval_policy.py --run-dir runs/k_hardmine --episodes 16 --seconds 20
```
Runs many randomized episodes (reset jitter, friction, and — for terrain runs
— mass/terrain/pushes) in parallel and reports fall rate, speed tracking, and
drift, sweeping the trained command/terrain range automatically. **Judge any
change this way, not from a single episode** — single scripted-gait runs
early in this project swung between "walks 0.5 m" and "falls over" for
neighboring parameters, purely from randomness.

```bash
python gait_stats.py --run-dir runs/k_hardmine --target-speed 0.8 --terrain-amplitude 0.12
```
Per-foot step rate, duty cycle, and swing height/timing — catches a lopsided
gait (e.g. front legs stepping twice as often as rear) that fall rate and
speed alone hide.

```bash
python play.py --run-dir runs/k_hardmine --target-speed 0.8 --terrain-amplitude 0.12 --record out.gif
```
Watch it (or record a GIF, avoiding the GLFW viewer entirely). `--run-dir`
rebuilds the exact env the checkpoint was trained with (kp/kd, reward
weights) from that run's saved `env_kwargs.json`.

## How the task is set up

- **Action space**: 12 target joint-position offsets (one per hip/thigh/
  calf joint), PD-tracked to torque at 250 Hz internally.
- **Observation** (51-dim): gravity vector in body frame, base angular
  velocity, base orientation quaternion, 12 joint positions (relative to
  a neutral standing pose), 12 joint velocities, the previous action, a
  3-dim velocity command (`[target_speed, 0, 0]`, sampled per episode when
  training with a speed range), and a 2-dim sin/cos gait-phase clock.
  The policy is **blind** — no terrain information (heightmap, ray casts) —
  it only feels the ground through joint torques and the trunk IMU.
- **Reward**: forward-velocity tracking (body-frame, if `--body-frame-
  velocity`) + zero-sideways-velocity tracking + upright orientation +
  heading/lateral-position drift penalties + torque/energy cost +
  action-rate smoothness + a diagonal-trot timing bonus against a
  speed-scaled clock (`--phase-match-weight`) + a capped foot air-time
  bonus + foot-clearance bonus + survival bonus. Optional but currently
  unused (0 weight) in the working recipe: trot-symmetry bonus and
  per-foot duty-cycle penalties — see `train.py --help` for why. Weights
  are in `Go1FlatEnv._compute_reward`.
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
- **Rough terrain** (`--terrain-amp-max`): a MuJoCo heightfield, regenerated
  every episode (interpolated random noise, feature size randomized 15 cm-1
  m, flat start pad, amplitude sampled per episode and ramped by a
  curriculum). Only built into the model when enabled — the flat env is
  bit-identical to before terrain support existed.

## Current status

16-64 randomized 20 s episodes per cell, deterministic policy, `runs/k_hardmine`:

| Terrain amplitude | Falls @ 0.3 m/s | Falls @ 0.8 m/s | Falls @ 1.0 m/s |
|---|---|---|---|
| flat (0 cm) | 0% | 0% | — |
| 4 cm | 0% | 0% | — |
| 8 cm | 0% | 0% | — |
| 12 cm | 0% | 0% | 0% |

Speed tracking holds within ~5-10% of every commanded speed at every
amplitude; the trot stays symmetric (all four feet at the same step rate and
duty cycle) throughout, with swing height rising automatically on rougher
terrain (6-10 cm on flat ground vs. 14-22 cm on 12 cm terrain at 0.8 m/s).

Getting here took several false starts worth knowing about if you're
extending this:
- **gSDE exploration nearly killed RL entirely.** Its effective noise
  (`std × 128-dim latent features`) toppled an untrained policy in ~0.7 s,
  so the first PPO runs only ever learned to lunge and fall. `train.py`
  now defaults to plain Gaussian noise, off by default (`--no-use-sde`).
- **A fixed gait-timing clock at full strength from step 0 makes things
  worse, not better** — it demands precisely-timed lifts before the policy
  can balance at all. Ramp it in instead (`--phase-match-warmup-steps`).
- **An uncapped air-time reward let one leg hover for 457 ms** and out-earn
  several correct steps elsewhere, causing a lopsided gait where front legs
  stepped twice as often as rear. Fixed by `--air-time-cap`.
- **Reward-shaping weights can be gamed.** A heavier trot-symmetry weight,
  tried alongside other shaping changes, converged to a hobble on a single
  diagonal pair (one pair almost always down, the other almost always up) —
  it satisfied the letter of "diagonal pairs alternate" without producing a
  real trot.
- **A terrain curriculum that only ramps the amplitude's upper bound
  under-trains the hardest corner** even after the ramp finishes, since
  sampling is still uniform from 0. Fixed with a `--terrain-amp-min`
  hard-mining fine-tune once the full curriculum has already run.

## Next stages

Ideas for extending past `runs/k_hardmine`, roughly in order of effort:

1. **Terrain-aware observation** — give the policy some exteroception (a
   small local heightmap or a handful of ray-cast height samples ahead of
   each foot) instead of pure proprioception. Likely the highest-leverage
   change for anything harder than the current terrain, since right now the
   policy only reacts to a bump after a foot has already landed on it.
2. **Slopes** — tilt terrain patches or ramp geoms at increasing angles
   (5°→20°); add a slope-angle term to the observation.
3. **Stairs** — stacked box geoms with step heights around the Go1's
   quoted 10 cm default step-climbing capability, then push beyond it.
4. **Discrete obstacles / gaps** — random box/cylinder clutter and narrow
   gaps, forcing more deliberate foot placement (pairs well with #1).
5. **Higher top speed** — the current trot always keeps ≥2 feet down; a
   faster gait needs a flight phase (0 feet down briefly), which the
   `phase_match` clock's stance/swing split would need to change to allow.
6. **Sim-to-real** — swap in the higher-fidelity mesh model from MuJoCo
   Menagerie (see the model note above), and add the usual sim-to-real
   staples: actuator/observation latency, torque-domain randomization
   (not just PD gains), and observation noise.

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
