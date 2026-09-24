# Teaching a Go1 to Walk: PPO, Reward Design, and the Project Journey

This is a companion piece to [README.md](README.md). The README is the
operational reference (how to run things, current results, what's next).
This document is the narrative: what PPO is and why it fits this problem,
how it was wired up to the Go1, how the reward function was actually built
up piece by piece (with the failures that motivated each piece), and the
methodology lessons that repeated across every stage. Wherever a number is
quoted, it's from a real logged run, not a guess.

**Contents**
1. [What PPO is, briefly](#1-what-ppo-is-briefly)
2. [Why PPO for this problem](#2-why-ppo-for-this-problem)
3. [Wiring PPO to the Go1](#3-wiring-ppo-to-the-go1)
4. [The reward function: built one failure at a time](#4-the-reward-function-built-one-failure-at-a-time)
5. [The staged curriculum: flat → speed → terrain → slopes → stairs → mesh](#5-the-staged-curriculum-flat--speed--terrain--slopes--stairs--mesh)
6. [Methodology lessons that kept recurring](#6-methodology-lessons-that-kept-recurring)
7. [Demo](#7-demo)

---

## 1. What PPO is, briefly

Proximal Policy Optimization (Schulman et al., 2017) is a policy-gradient
reinforcement learning algorithm. The setup is standard RL: an **agent**
(here, a neural network policy) observes a **state** from an
**environment** (here, the Go1's joint angles, velocities, orientation,
etc. in MuJoCo), picks an **action** (here, 12 target joint-angle offsets),
and receives a scalar **reward** each step. The goal is to find policy
parameters that maximize the expected sum of future rewards.

Plain policy-gradient methods (REINFORCE, vanilla actor-critic) update the
policy directly in the direction that increases the probability of
actions that led to high reward. This works but is fragile: a single large
update can push the policy into a bad region of parameter space it can't
recover from, especially with noisy, high-variance gradient estimates —
exactly what you get from a few hundred parallel physics simulations.

PPO's fix is deceptively simple. It still computes a policy-gradient-style
update, but **clips** how far the new policy is allowed to move from the
old one in a single step:

```
L_CLIP(θ) = E[ min( r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A ) ]
```

where `r(θ) = π_θ(a|s) / π_θ_old(a|s)` is the probability ratio between
the new and old policy for the action actually taken, `A` is the
**advantage** (how much better this action was than the policy's own
baseline expectation, estimated with Generalized Advantage Estimation —
GAE), and `ε` (`clip_range` in code, `0.2` throughout this project) bounds
the ratio. If an update would move the policy so far that `r(θ)` leaves
`[1-ε, 1+ε]`, the objective clips it — no incentive to keep pushing
further in that direction *for that batch*, without an expensive
second-order trust-region computation (PPO's predecessor, TRPO, solved the
same problem but needed one). The result is an algorithm that's simple to
implement, cheap to run, and — critically for physical simulation — stable
enough to survive being pointed at a system as prone to catastrophic,
irrecoverable failure (falling over) as a legged robot.

PPO is **on-policy**: it collects a batch of fresh experience under the
current policy, does a small number of gradient epochs over that batch
(`n_epochs=10` here), then throws the batch away and collects a new one.
It's also **actor-critic**: alongside the policy ("actor"), it trains a
value function ("critic") that predicts expected future reward from a
state, used to compute the advantage estimates that drive the policy
update. Both networks here are small MLPs, `[256, 256, 128]` for each
(shared architecture, no shared weights) — plenty for a 51-63-dimensional
observation, more would just cost wall-clock time for no benefit at this
problem scale.

## 2. Why PPO for this problem

- **Continuous action space.** The Go1 has 12 actuated joints; actions are
  continuous target-angle offsets, not a discrete menu of moves. PPO
  handles continuous actions natively via a Gaussian policy (mean from the
  network, learned or fixed log-std) — no discretization needed.
- **Physics sim is your training data generator, and it's not free.**
  Every environment step means running MuJoCo's contact solver. PPO's
  on-policy, batch-collect-then-update structure parallelizes cleanly
  across many simultaneous simulated robots (`SubprocVecEnv`, 16 envs
  here, each an independent MuJoCo instance), which is how you get enough
  samples per wall-clock hour to make locomotion RL practical at all —
  every run in this project used 16-way parallelism, at roughly
  2000-2800 simulated steps/second combined.
- **Robustness to reward-shaping churn.** This project's reward function
  changed dozens of times (Section 4). Off-policy methods (SAC, TD3) reuse
  old transitions from a replay buffer under an old reward definition,
  which becomes stale or actively wrong the moment the reward changes.
  PPO's on-policy, throw-the-batch-away design means every reward change
  takes effect immediately and cleanly on the very next rollout.
- **It's the standard for this exact problem.** Modern sim-to-real
  quadruped locomotion research (Rudin et al. 2022 "Learning to Walk in
  Minutes", Hwangbo et al. 2019, and most of what followed) converged on
  PPO plus reward shaping plus domain randomization as the default recipe.
  This project follows that lineage explicitly — several reward terms
  below are direct descendants of that literature, adapted by trial and
  error to this specific model and reward mix.

## 3. Wiring PPO to the Go1

**Observation** (51 dims, or 60 with `--use-terrain-heightmap`; all built
in `Go1FlatEnv._get_obs()`):

| dims | content |
|---|---|
| 3 | gravity vector in the trunk's own frame (tells the policy which way is "up" without giving it absolute orientation) |
| 3 | angular velocity (gyro) |
| 4 | base orientation quaternion |
| 12 | joint angles, relative to a fixed default standing pose |
| 12 | joint velocities |
| 12 | the previous action (helps the policy reason about its own action-rate penalty and produce smooth output) |
| 3 | commanded velocity (vx, vy, yaw-rate — only vx is actually used/randomized in this project) |
| 2 | a gait-phase clock, `(sin, cos)` of a periodic phase variable |
| +9 (optional) | a 3×3 local heightmap ahead of the trunk (Section 5, stage 3d) |

Everything is normalized on top of this by `VecNormalize` (running
mean/std per dimension, clipped to ±10), which matters more than it might
look — without it, dimensions on wildly different scales (joint velocity
in rad/s vs. a unit gravity vector) make the policy's first-layer weights
have to compensate for scale before they can even start learning anything
about the actual signal.

**Action** (12 dims, `Box(-1, 1)`): *not* raw motor torques. Each action
dimension is a target joint-angle offset from a fixed default pose,
scaled by `action_scale` (0.5 by default) and PD-tracked:

```python
target_qpos = DEFAULT_JOINT_POS + action_scale * action        # policy's "intent"
target_qpos = clip(target_qpos, joint_range)
torque = kp * (target_qpos - q) - kd * dq                      # PD control, in Python
torque = clip(torque, torque_range)
```

This is the standard choice in the sim-to-real quadruped literature (the
Rudin et al. style referenced in the code), and for a good reason: asking
the network to output raw torque means it has to *learn* PD control from
scratch (relearning "if the joint overshoots, push back") on top of
learning to walk — throwing away a free, well-understood control primitive
for no benefit. Target-position-plus-PD gives the policy a much shorter,
more forgiving path from "random initial weights" to "something that
doesn't immediately fall over," because even a randomly-initialized policy
outputting near-zero actions produces a stable standing pose, not chaotic
torque.

**Domain randomization**, layered in progressively across the project (see
Section 5): floor/foot friction, trunk mass, random pushes, terrain shape,
and (later, opt-in) PD-gain/torque-scale variance, action/observation
latency, and observation noise for sim-to-real robustness.

**Curriculum learning**, via a small generic `RampCallback`: linearly ramps
one env attribute (e.g. the upper bound of sampled terrain amplitude) from
a start value to a final value over a configured number of steps, logged
to TensorBoard as `curriculum/<name>`. Every new terrain/speed/difficulty
axis in this project was introduced this way on its first appearance —
except once (Section 6), which is exactly when things broke.

## 4. The reward function: built one failure at a time

The final reward (`Go1FlatEnv._compute_reward`) sums 18 named components.
None of them were designed up front as a complete package — every one
exists because an earlier, simpler version produced a specific, observable
pathology. This section is that history, roughly in the order terms were
added.

### The two bugs that had nothing to do with reward shaping

Before any reward tuning could mean anything, two bugs made every single
early RL run fail outright, regardless of reward design:

1. **Foot contact detection.** Foot-contact was checked against a foot
   *site*'s height threshold (0.015 m), but the site sits at the physical
   sphere's *center*, so a foot resting on the ground actually reads
   0.022 m — above the threshold. Every contact-dependent reward term
   (duty cycle, clearance, air-time, gait pattern, trot symmetry) was
   silently reading a standing robot as having **zero feet down**, all the
   time. Fixed by switching to real MuJoCo contact forces
   (`force > 1 N`), which also turned out to be necessary for rough
   terrain later (a foot on a heightfield can register multiple contact
   points; a height threshold can't handle that at all).
2. **Exploration noise was destructive.** The first attempts used gSDE
   (generalized State-Dependent Exploration) at `log_std_init=0`. gSDE's
   noise scales with a 128-dimensional latent feature vector, and at that
   setting it was violent enough to knock the untrained robot over in
   about 0.7 seconds — before it could ever experience a single reasonable
   step of standing, let alone walking. Measured with plain per-action
   Gaussian noise instead: survival time was 1.1s at `std=1.0`, 4s at
   `0.37`, 19s at `0.22`. The fix was mundane once found: no gSDE, and a
   conservative fixed initial log-std (`-1.0` to `-1.5`, i.e. std
   `0.37`–`0.22`) so the policy could experience *some* successful
   standing before exploration noise had to teach it anything else.

Neither of these is a reward-shaping problem, but they illustrate
something that held throughout the whole project: a policy that "won't
learn" is often not making a subtle reward-weighting mistake — it's
failing to ever experience the successful behavior often enough for any
reward signal to reinforce it. Diagnosing *that* took inspecting actual
contact forces and actual survival times, not more reward-tuning guesses.

### Base shaping: get it moving, keep it upright

- **`r_velocity`** — Gaussian reward for forward-velocity tracking,
  `exp(-2 · normalized_error²)`, where the error is normalized by
  `target_speed` itself, not used as a raw absolute error. This one detail
  mattered a lot: with an *absolute* error, a low `target_speed` (e.g.
  0.2 m/s) meant standing still (a small absolute error) scored almost as
  well as actually walking — 0.923/1.0 in one measured case — which
  removed most of the incentive to move at all. Normalizing keeps
  "standing still" penalized to the same *relative* degree regardless of
  the commanded speed.
- **`r_orientation`** — penalizes the trunk's roll/pitch (via the
  body-frame gravity vector's x/y components, which should be ~0 when
  upright). Blind to yaw by construction (a body-frame gravity vector
  doesn't change under pure yaw rotation), which is exactly why heading
  drift (below) needed its own separate term.
- **`r_torque`**, **`r_action_rate`**, **`r_ang_vel`** — small
  regularization penalties (energy cost, output smoothness, angular
  velocity) present from early on to keep the policy from finding jerky,
  high-frequency, unrealistic solutions that happen to satisfy other terms.
  `r_action_rate`'s weight was raised from an original `0.01` after a
  converged policy exploited a weak version of it: it satisfied the
  contact-count and trot-symmetry terms with tiny, fast corrections
  instead of a bold, visible stride — "small fast shuffling" instead of
  walking.
- **`r_alive`** — a flat per-step survival bonus, deliberately kept small
  (`0.2`, down from an original `0.5`). At `0.5`/step over a 500-step
  episode, this alone was large enough to plausibly dominate the whole
  incentive calculus — a policy that plays maximally safe collects nearly
  all of it "for free," while actually trotting risks losing it by
  falling. Present, but no longer overwhelming the gait-quality terms.
- **`r_height`** — penalizes trunk height dropping below ~0.30 m
  (terrain-relative, not absolute — this had to be fixed later once
  rough terrain existed at all).

### Drift: velocity-based shaping wasn't enough

- **`r_lateral`** penalizes instantaneous sideways/vertical velocity, and
  was present early — but it wasn't enough on its own. A policy can carry
  a small, nearly-free constant yaw rate that barely registers in a
  per-step angular-velocity penalty, yet compounds into meters of heading
  drift over a 20-second episode. This is literally what "walks fine but
  veers right" looked like at the per-step level: nothing *looked* wrong
  in any single-step reward breakdown. The fix was **`r_heading`**,
  penalizing actual accumulated yaw and y-position deviation from the
  start pose directly, not just the instantaneous rate — because the
  problem was in the integral, not the derivative.
- **`r_lat_track`** (`lateral_tracking_weight`) — a Gaussian bonus
  specifically for near-zero lateral velocity, added later as an
  additional, more tunable lever on top of `r_lateral`'s raw quadratic
  penalty; useful because its width (`lateral_tracking_sigma`) is a single
  interpretable knob, whereas raising `r_lateral`'s fixed weight can't be
  tuned independently.
- **`r_yaw_rate`** — direct yaw-rate penalty, off (`0.0`) by default since
  `r_heading` already targets the actual failure mode more precisely for
  straight-line walking; exists for when a nonzero commanded yaw-rate is
  ever wanted.
- One reward change alone did **not** fix a specific crabbing failure
  (`runs/f_shaped`, which walked sideways ~1.45 m off course in 20s on a
  diagonal-pair-hobbling gait) — the actual fix there was switching to
  **body-frame velocity** (`body_frame_velocity=True`, rotating world-frame
  velocity into the trunk's own frame before computing `r_velocity`), so
  "forward" always means "the direction the robot itself is facing," not
  a fixed world axis it might have already turned away from.

### Gait quality: from "any 2 feet down" to a specific, prescribed pattern

Four terms layer on top of each other here, each addressing a way the
previous one alone could still be gamed:

- **`r_gait`** rewards exactly 2 feet in contact (`+0.08`), is neutral at 3
  (sloppy but not dangerous, `0.0`), and specifically penalizes exactly 1
  foot down (`-0.15`, worse than 0 or 4 feet at `-0.05`) — added after
  directly observing a fall from precisely that configuration: three legs
  simultaneously airborne, single-point support, no roll stability at all.
  Before this distinction, 1-foot and 3-foot support scored identically
  (both neutral), giving no specific incentive to avoid the far more
  dangerous case.
- **`r_foot_duty`** tracks each foot's own EMA'd (decay 0.98, ~1s time
  constant) contact fraction and penalizes it for exceeding
  `max_foot_duty_cycle` or falling below `min_foot_duty_cycle`. This is
  what actually stops one specific foot from "opting out" of the gait
  permanently (e.g. rear legs that just drag) — unlike the count-based
  terms above, it evaluates *every* foot on *every* step regardless of
  what's currently swinging, so there's no way for a foot to simply never
  be checked. This term exists because the count-based terms alone
  produced exactly that failure once: `runs/e_long_d`'s rear legs stepped
  at half the rate of the front legs (1.45 vs 2.9 steps/s).
- **`r_phase_match`** goes further still: rather than any loose "some
  acceptable split," it prescribes an *exact* footfall timing (trot:
  diagonal pairs FR+RL / FL+RR alternate; a "bound" gait style — front
  pair / rear pair — was also implemented and tried, see Section 5) via a
  periodic phase clock, and rewards matching it. An unambiguous target is
  much easier for per-step Gaussian action noise to discover than "find
  some good rhythm from scratch."
- **`r_trot_symmetry`** rewards the two diagonal pairs actually agreeing
  internally *and* alternating with each other, but only when exactly 2
  feet are down — guarded specifically because, without that guard, a
  frozen, all-four-feet-planted pose trivially satisfies "pairs agree" and
  scored close to full marks for literally standing still, which
  contributed to the policy collapsing into a frozen pose once training
  stabilized.
- **`r_foot_clearance`** rewards a *swinging* foot for lifting toward
  `target_clearance` (later made adaptive to stair height, Section 5).
  None of the terms above care *how high* a foot lifts mid-swing, only
  whether it's touching — so without this, tiny near-ground shuffling
  satisfies every other gait term exactly as well as a bold, visible
  stride, and there's no pressure to prefer the latter once the former is
  found.
- **`r_air_time`** (Rudin et al. 2022 style): on the exact step a foot
  touches back down, reward it for how long it had just been airborne,
  relative to `target_air_time` — deliberately the *only* gait term that
  never specifies exactly *when* a foot should swing, only that a
  completed swing was a reasonable duration. This leaves the policy free
  to find its own step frequency rather than matching a fixed clock picked
  without empirical grounding. It also caused its own bug once: an
  uncapped version let a policy get extra reward for one foot swinging
  unusually long, producing a visible **limp** — fixed by `air_time_cap`,
  which caps the credited air time at `target_air_time` (rewarding "long
  enough," not "as long as possible").
- **`r_static_stability`** — the newest term, added in stage 3e (below):
  directly rewards *more than 2* feet down, scaled by how tall the current
  stair riser is. Unlike every term above (which all implicitly reward or
  tolerate the standard always-2-feet-down trot), this is a deliberate
  attempt to pull the policy toward a different support pattern
  specifically when a stair demands it — tried only after three other,
  structurally different fixes for the same problem had already failed
  (see Section 5's stairs write-up).

## 5. The staged curriculum: flat → speed → terrain → slopes → stairs → mesh

The reward function above didn't arrive all at once — it grew stage by
stage, alongside a terrain/speed curriculum that also grew stage by stage.
Every stage followed roughly the same pattern: introduce one new axis of
difficulty (gradually, via `RampCallback`), evaluate over many randomized
seeds (never trust a single episode — see Section 6), find the specific
failure mode, fix it, re-verify nothing else broke.

- **Stage 0 — scripted crawl, no RL.** Before any learning, a hand-scripted
  gait confirmed the physical model itself could walk at all. The
  first attempt didn't: it moved all four legs but achieved ~0 net forward
  motion, traced to the center of mass falling outside the support
  triangle during rear-leg swings — a foundational insight (weight-shift
  before lifting a leg) that resurfaced later, almost verbatim, as the
  hypothesis for the stairs limit.
- **Stage 1a — flat, fixed 0.3 m/s** (`runs/e_long_d`, then `h_clock`).
  First working RL walk: 0% falls, but asymmetric (rear legs at half the
  front legs' step rate) and, once a fixed gait-phase clock was added to
  force the trot pattern, still limited to almost exactly the one speed it
  was trained at (0.086 m/s achieved when *commanded* 0.5 m/s).
- **Stage 1 — speed curriculum, 0.2–1.0 m/s** (`runs/i_speed_curriculum`,
  fine-tuned from `h_clock`): ramping the *commanded* speed range during
  training (not just training at one fixed speed) fixed the
  generalization gap — 0% falls at every commanded speed 0.3–1.0 m/s, only
  ~10% short of target at the very top.
- **Stage 2 — rough terrain** (`runs/j_terrain`): heightfield bumps
  (0→12cm curriculum), floor friction 0.5–1.25×, trunk mass ±10%, random
  pushes. 0% falls almost everywhere, but 6% falls at the single hardest
  combined corner (12cm bumps + 0.8 m/s). Root cause, found by inspecting
  fall traces directly: the terrain curriculum only ramped the sampled
  amplitude's *upper bound*, so most training episodes still sampled from
  the easy end of the range even after the ramp finished — the truly hard
  corner was a thin slice of what the policy actually experienced. Fixed
  (`runs/k_hardmine`) by biasing the resampled training distribution
  directly at the hard region (narrow amplitude *and* speed ranges,
  curricula skipped) for a short fine-tune: 4%→**0%** at that corner, with
  no regression anywhere else in the tested grid. This became a reusable
  lesson (Section 6).
- **Stage 3a — slopes** (`runs/l_slopes` → `m_slope_hardmine` →
  `n_slope_steep_hardmine` → `o_slope_consolidate`): took zero-shot
  tolerance (~5–10°) up to 0% falls at every angle to ±20°, but only after
  three narrow hard-mining fine-tunes kept trading one corner's quality
  for another's — a **broad** consolidation pass (train on the *entire*
  range at once, with less reopened exploration noise) is what actually
  resolved it cleanly, the opposite lesson from the terrain-amplitude fix
  above. Left one accepted trade-off: flat-ground high-speed drift
  regressed slightly (~0.6–0.7m) as the cost of across-the-board slope
  robustness.
- **Stage 3b — stairs** (`runs/p_stairs`): clean to ±6cm, but ascending
  stairs above ~8–10cm produces a genuine **stuck-without-falling**
  failure — the robot plateaus at the first or second step and oscillates
  there for the rest of the episode. This is *not visible in fall-rate
  metrics at all* (the robot doesn't fall — fall rate looked fine), and
  was only found by the user's own manual/visual testing. Four
  structurally different fixes were tried, in order: an adaptive
  foot-clearance target (fixed a real 4cm-cap bug, no behavior change), a
  stair-height-scaled gait clock (more swing time, no behavior change), a
  terrain-aware local-heightmap observation (Section "network surgery"
  below — genuinely helped moderate heights, made the extreme case
  *worse*), and a non-trot static-stability reward (Section 4's
  `r_static_stability` — measurably nudged behavior in the right
  direction, but not enough to change the outcome). **All four hit the
  identical stall position.** That consistency across four unrelated
  intervention types is itself the evidence: it's a local optimum of
  fine-tuning an already-converged, trot-locked policy (which structurally
  never leaves a 2-feet-down support pattern), not a missing reward
  ingredient — consistent with the Go1's own ~10cm rated step-climbing
  spec. Accepted as a documented limit rather than chased further; a real
  fix would likely mean training stairs in from the start (before the trot
  locks in) rather than fine-tuning on top of it.
- **Network-surgery warm-starting** (`expand_obs_checkpoint.py`): to test
  the heightmap-observation idea above without retraining the entire
  staged lineage from scratch, a small tool copies every weight of an
  existing policy except the very first layer, zero-initializing the new
  input columns needed for the larger observation. The expanded model is
  mathematically *identical* to the original at t=0 (verified: matched-seed
  observations and actions checked bit-for-bit against the original before
  trusting any fine-tune built on it) — fine-tuning then discovers whether
  and how to use the new signal, without paying for tens of millions of
  steps of already-learned behavior a second time.
- **Stage 3c — discrete obstacles**: turned out to already be solved
  zero-shot (0% falls to 14cm, only 0–6% at 18–22cm — taller than the
  robot's own thigh segment) once a real placement bug was fixed (obstacles
  were originally scattered across the full course width, but the robot
  only wanders ±0.4–0.5m off centerline in practice, so most obstacles
  never actually crossed its path — the first "0% falls" result was
  trivially true, not actually informative).
- **Stage 3f — sim-to-real robustness** (torque/PD-gain randomization,
  action/observation latency, observation noise): the zero-shot baseline
  was already robust to torque/PD variance and noise, but showed a real
  37.5%-fall vulnerability to control-loop latency. The first fine-tuning
  attempt introduced all four new randomization axes at *full* configured
  strength from step 0 — the only axis in the entire project skipped a
  gradual curriculum on first introduction — and broadly regressed
  everything, including the latency metric it targeted. A fix
  (`sim2real_scale_current`, a shared 0→1 curriculum multiplier scaling
  every axis's deviation from nominal) corrected the methodology but
  produced an almost *identical* regression, with latency's fall rate
  landing at exactly 37.5% again. Working theory: the policy has no
  observation channel for its own episode's actual latency, so it can't
  specialize per-episode and instead settles on one compromise that's
  worse everywhere. The real fix (teacher/student distillation with
  privileged latency information during training) is a materially bigger
  technique, not attempted — accepted as a documented limit, the same way
  as the stairs stall.
- **Stage 4 — higher-fidelity mesh model**: swapped the original
  primitive-geometry model (boxes/capsules, built from published reference
  dimensions) for the official MuJoCo Menagerie mesh model (real meshes,
  per-link inertial tensors, per-link collision geometry, real joint
  ranges), changed as *one isolated variable* — solver settings, PD gains,
  and actuator type were all deliberately kept identical so any behavior
  change could be attributed to geometry/inertia alone. Zero-shot: no new
  falls anywhere in the whole previously-tested grid, and the ascending-
  stairs stall reproduced almost exactly (0.084 vs 0.091 m/s) — useful
  independent confirmation that it's a real gait-strategy limit, not an
  artifact of the old simplified geometry. Zero-shot did reveal a
  consistent ~15–30% speed-tracking overshoot (real inertia changes how
  the same PD torques convert to velocity) and worse downhill-slope drift;
  an 8M-step fine-tune (full terrain/slope/stair/speed ranges active from
  the start, no re-curriculum needed) fixed both, leaving only a small,
  confirmed-not-noise increase in the already-hardest existing corner
  (descending 12cm stairs: 4%→8%→12% falls across
  original→zero-shot→fine-tuned).

## 6. Methodology lessons that kept recurring

These weren't planned in advance — they're patterns that showed up more
than once, expensively enough the first time to be worth naming.

- **Fall rate alone can completely miss a failure.** A policy that stalls
  without tipping over reports a *low, reassuring* fall rate while
  totally failing the task (stage 3b). Always sanity-check a policy
  visually (or by tracing trunk position over time), not just by
  pass/fail statistics.
- **Ramping only a range's upper bound under-trains the hard corner, even
  after the ramp finishes**, if sampling stays uniform across the whole
  range (stage 2's terrain-amplitude bug). Either raise the lower bound
  too, or do a short hard-mining fine-tune biased at the hard region once
  the main curriculum completes.
- **Narrow hard-mining can trade one corner's quality for another's,
  repeatedly** (stage 3a's three-fine-tune slope saga) — if the *same*
  narrowing trick needs a third application, that's the signal to try the
  opposite: a broad pass over the *entire* range at once, with less
  reopened exploration noise. This is not universal, though — the same
  broad-consolidation fix that cleanly worked for slopes made stairs
  *worse* on gait symmetry and flat-ground drift (stage 3b) without fixing
  its actual problem. Never assume a fix that worked for one terrain type
  transfers to another; re-verify every time.
- **Introduce every new randomization/difficulty axis with a gradual
  curriculum on its first appearance — no exceptions.** The one time this
  didn't happen (stage 3f, sim-to-real robustness axes at full strength
  from step 0) produced a broad regression that a fix to the actual
  mechanism (a shared curriculum scale) only partially undid.
- **Check a run's own checkpoint history before assuming a fresh design
  flaw.** A stage-3a regression was traced to the *exact* point the slope
  curriculum first reached its maximum, not gradually — evidence of an
  under-trained corner (fixable by more training there) rather than a
  wrong reward design (which would need a redesign).
- **Multi-seed evaluation, not single episodes, for anything that matters**
  (this predates RL entirely — the scripted stage-0 gait already taught
  this lesson: single deterministic runs were chaotic enough that
  neighboring parameters flipped between "+0.5m" and "falls over"). Every
  reported number in this project and in this document comes from
  multiple randomized seeds (typically 8–24 twenty-second episodes),
  never one run.
- **A clean "0% falls" result is only meaningful once you've confirmed the
  hard part of the terrain was actually in the robot's path** (stage 3c's
  obstacle-placement bug) — otherwise it can be trivially true and tell
  you nothing.
- **When testing signed terrain parameters, always include the decimal
  point** (`--stair-height -0.06`, not `-6`) — a bare small integer is
  silently interpreted as *metres*, producing absurd terrain and
  misleadingly identical results across different inputs. This happened
  once; the tell was identical stats for supposedly different settings.

## 7. Demo

Two recorded GIFs of `pretrained/x_mesh_finetune` (the current
higher-fidelity mesh-model checkpoint, Section 5's Stage 4) are committed
in [`media/`](media/) and shown in the [README](README.md#stage-4-higher-fidelity-mesh-model):

<p float="left">
  <img src="media/mesh_model_flat.gif" width="380" alt="Go1 walking on flat ground">
  <img src="media/mesh_model_slope_descent.gif" width="380" alt="Go1 descending a 20 degree slope">
</p>

*Left: flat ground, 0.5 m/s. Right: descending a 20° slope at 0.3 m/s.*

To watch any checkpoint yourself, or generate a new GIF:

```bash
# Live viewer (needs a display / GLFW):
python play.py --run-dir pretrained/x_mesh_finetune --target-speed 0.5

# Or render offscreen straight to a GIF, no window needed:
python play.py --run-dir pretrained/x_mesh_finetune --target-speed 0.5 \
    --record my_demo.gif --camera track
```

Useful variations (see `python play.py --help` for the full list):

```bash
# Try the original (non-mesh) checkpoint on stairs:
python play.py --run-dir pretrained/p_stairs --stair-height 0.06 --record stairs.gif

# Watch the mesh model handle rough terrain at a specific amplitude/speed:
python play.py --run-dir pretrained/x_mesh_finetune --terrain-amplitude 0.12 \
    --target-speed 0.8 --record rough.gif

# Reproducible episode (same seed -> same random terrain/pushes/friction):
python play.py --run-dir pretrained/x_mesh_finetune --seed 5 --record repeat.gif

# chase_rear (from behind) is often more informative than the default
# track (side-on) camera for spotting left/right asymmetry or a stair
# approach head-on; topdown is best for footfall-timing patterns.
python play.py --run-dir pretrained/p_stairs --stair-height 0.12 \
    --camera chase_rear --record stairs_rear.gif
```

To numerically evaluate a checkpoint over many randomized seeds (the
actual methodology behind every number in this document — see Section 6):

```bash
python eval_policy.py --run-dir pretrained/x_mesh_finetune --episodes 16
```

`--frame-stride N` (default 2, used at 4 for the GIFs above) keeps every
Nth rendered frame, roughly halving file size per doubling of `N` — worth
raising for anything meant to be committed to the repo or shared, since
the default is tuned for closer visual inspection rather than small files.
