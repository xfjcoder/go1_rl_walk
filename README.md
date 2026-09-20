# Go1 Quadruped Locomotion — Stage 1: Flat Terrain

Trains a Unitree Go1 to walk forward using MuJoCo physics + PPO
(Stable-Baselines3). This is the first stage of a terrain curriculum —
the env is structured so later stages (rough ground, slopes, stairs)
drop in without touching the reward or observation code.

```
go1_rl_walk/
├── assets/go1.xml       # MJCF model (self-contained, capsule-based)
├── envs/go1_env.py       # Gymnasium env: obs, action, reward, termination
├── configs/flat_terrain.yaml  # hyperparameters / reward weights (reference)
├── train.py               # PPO training entrypoint
├── play.py                # Load a checkpoint and watch it walk
├── smoke_test.py           # Sanity-check the model with no RL deps
└── requirements.txt
```

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate   # or conda
pip install -r requirements.txt
```

GPU is optional for MuJoCo (CPU physics is fine), but strongly speeds up
PPO's neural-net updates. A modern laptop CPU can still train this in a
few hours with `--n-envs 8`.

## 2. Sanity check the model first

```bash
cd go1_rl_walk
python smoke_test.py --view
```

This loads `assets/go1.xml`, holds the standing pose with a simple PD
controller (no RL yet), and opens the MuJoCo viewer so you can confirm
the robot stands without exploding/clipping through the floor. Fix any
model issues here before spending compute on training.

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

## 3. Train

```bash
python train.py --timesteps 20000000 --n-envs 16
```

- Checkpoints save to `checkpoints/` every ~1M steps.
- Progress logs to `logs/` — watch live with:
  ```bash
  tensorboard --logdir logs/
  ```
  Track `rollout/ep_rew_mean` (should climb steadily) and, if curious,
  add custom scalars from `info["reward_components"]` for per-term
  debugging.
- Resume an interrupted run:
  ```bash
  python train.py --resume checkpoints/go1_flat_10000000_steps.zip
  ```

**Expected timeline** (see `configs/flat_terrain.yaml` for the fuller
breakdown): flailing for the first ~1M steps, stable standing by ~2-3M,
a recognizable trot by ~5-12M, a smooth, robust gait by ~20M.

## 4. Watch the trained policy

```bash
python play.py --model checkpoints/go1_flat_final.zip
```

## How the task is set up

- **Action space**: 12 target joint-position offsets (one per hip/thigh/
  calf joint), PD-tracked to torque at 250 Hz internally. This is far
  more sample-efficient than commanding raw torques directly.
- **Observation** (49-dim): gravity vector in body frame, base angular
  velocity, base orientation quaternion, 12 joint positions (relative
  to a neutral standing pose), 12 joint velocities, the previous
  action, and a 3-dim velocity command (currently fixed at
  `[1.0, 0, 0]` m/s — walk straight forward).
- **Reward**: forward-velocity tracking + upright orientation + low
  lateral drift + torque/energy cost + action-rate smoothness + a
  light gait-shaping term (discourages either 0 or 4 feet down at
  once) + survival bonus. Weights are in
  `Go1FlatEnv._compute_reward` / mirrored in `configs/flat_terrain.yaml`.
- **Termination**: trunk tips past ~60° from vertical, or trunk height
  leaves the `[0.15, 0.6]` m band (fell over / flew off the ground).
- **Domain randomization** (light, even on flat ground): small joint
  and orientation noise on reset, randomized floor friction. This
  alone measurably improves robustness and is a warm-up for the
  heavier randomization you'll want on rough terrain.

## Next stages: terrain curriculum roadmap

Once flat-terrain walking is solid (consistent forward velocity
tracking, doesn't fall over across many seeds), extend to:

1. **Rough terrain** — replace the flat `<geom type="plane">` floor
   with a heightfield (`<hfield>` + `<geom type="hfield">` in MuJoCo)
   generated from Perlin/simplex noise. Start with a shallow amplitude
   (±2 cm) and increase gradually — this is itself a curriculum.
2. **Slopes** — tilt terrain patches or add ramp geoms at increasing
   angles (5° → 20°); add a slope-angle term to the observation so the
   policy can condition on it.
3. **Stairs** — stacked box geoms with step heights around the Go1's
   quoted 10 cm default step-climbing capability, then push beyond it.
4. **Discrete obstacles / gaps** — random box/cylinder clutter and
   narrow gaps, forcing more deliberate foot placement.
5. **Terrain curriculum manager** — track per-episode success and
   auto-promote each parallel env to harder terrain as it succeeds
   (the standard approach in Rudin et al., "Learning to Walk in
   Minutes," and similar work), rather than training one terrain at a
   time.

For each new stage, you'll typically only touch: the MJCF terrain
block, the domain-randomization ranges in `reset()`, and possibly add
a terrain-encoding term to the observation. The action space, PD
control, and core reward shape carry over unchanged.

## Troubleshooting

- **Robot immediately collapses in smoke_test.py**: check `kp`/`kd`
  gains and the `stand` keyframe joint angles match your intended
  standing pose; also confirm the floor plane geom has nonzero
  friction.
- **Policy learns to stand but never walks**: increase
  `velocity_tracking` weight relative to `alive_bonus`, or reduce
  `alive_bonus` — it's easy for a lazy "just stand still" policy to
  dominate early training.
- **Jittery/high-frequency leg motion**: raise `action_rate_penalty`
  and/or lower `action_scale`.
- **Asymmetric/lopsided gait** (e.g. one leg taking large slow steps
  while the others take small fast steps): the base reward only cares
  about *how many* feet are on the ground, not *which* ones, so it
  doesn't by itself steer the policy toward a canonical diagonal trot
  — a lopsided-but-stable gait can score just as well. A
  `trot_symmetry` reward term (weight `--trot-weight`, default `0.15`)
  rewards the standard diagonal footfall pattern (FR+RL together,
  FL+RR together, the two pairs alternating). Bake this into a run
  from step 0 rather than introducing it mid-resume (see the entropy
  blowup case study below for why that combination is risky):
  ```bash
  python train.py --timesteps 20000000 --trot-weight 0.3
  ```
- **Policy degrades into near-random motion after a resume** (e.g.
  only one side of the robot moves, legs drag): check
  `train/std` in your log or TensorBoard. If it's climbed well above
  ~1.0 (action space is `[-1, 1]`, so std should normally stay under
  ~0.5), this is **entropy blowup**: the `ent_coef` entropy bonus for
  continuous actions is unbounded above, and if the policy-gradient
  signal goes quiet — which commonly happens right after changing a
  reward term (like `--trot-weight`) mid-resume, since the value
  function needs time to re-adapt — the entropy term can dominate the
  loss and drive `std` upward without limit. This actually happened
  during flat-terrain tuning here: an 8M-step resume combining
  `ent_coef=0.01` with a `trot-weight` change took `std` from ~0.45 to
  ~3.0, and the resulting policy only moved one side of the robot.
  `train.py` now includes a `StdGuardCallback` that auto-stops
  training if `std` exceeds `1.5`, so this costs a few thousand wasted
  steps instead of an entire run. Avoid it going forward by never
  changing `ent_coef` and a reward-shaping weight in the same resume —
  tune one at a time, in separate runs.
- **A good checkpoint got silently overwritten by a later bad run**:
  every run writes to `checkpoints/go1_flat_final.zip`, so a
  worse resume clobbers a better one under that name. `train.py` now
  also saves a uniquely-named `go1_flat_final_<total_steps>.zip`
  every time, which is never overwritten — check there if
  `go1_flat_final.zip` turns out to be worse than a version you had
  before. The numbered `go1_flat_<steps>_steps.zip` checkpoints saved
  periodically during training (every ~1M steps) are also never
  overwritten and are worth checking if a run degrades partway
  through.
- **Robot walks out of the camera frame**: the model includes a
  `track` camera (mode `trackcom`) that follows the trunk — both the
  interactive viewer and `--record` use it by default, so this
  shouldn't happen with the current files. If you're still seeing a
  static viewpoint, make sure you're on the latest version of
  `assets/go1.xml`.
- **Training is slow**: increase `--n-envs` (bounded by CPU cores),
  or reduce `n_steps`/`batch_size` for faster iteration at the cost of
  some sample efficiency.
